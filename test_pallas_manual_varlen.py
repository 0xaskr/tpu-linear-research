from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import jax
import functools
import numpy as np
import torch
import math

def cdiv(a: jax.Array, b: jax.Array | int):
  return jnp.ceil(a / b)

def cdiv_pt(a, b):
  return (a + b - 1) // b

def AlignUP(a, b):
  return (a + b - 1) // b * b

def prepare_chunk_offsets(seqlens: jax.Array, chunk_size:int):
  return jnp.pad(cdiv(jnp.diff(seqlens), chunk_size).astype(jnp.int32), (1, 0), constant_values=0).cumsum(-1)

def pad_to_multiple(x: jax.Array, multiple: int, axis: int, val):
  if multiple <= 1:
    return x
  shape = list(x.shape)
  length = shape[axis]
  remainder = length % multiple
  if remainder == 0:
    return x
  pad_len = multiple - remainder
  pad_width = [(0, 0)] * len(shape)
  pad_width[axis] = (0, pad_len)
  return jnp.pad(x, pad_width, constant_values=val)

def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generate chunk indices for variable length sequences.

    Example:
        >>> cu_seqlens = torch.tensor([0, 2, 5])
        >>> chunk_size = 2
        >>> prepare_chunk_indices(cu_seqlens, chunk_size)
        tensor([[0, 0],
                [1, 0],
                [1, 1]])
        Explanation:
        - Sequence 0 (len 2): 1 chunk (chunk 0)
        - Sequence 1 (len 3): 2 chunks (chunk 0, chunk 1)

    Returns:
        torch.LongTensor: A tensor of shape [Num_Total_Chunks, 2].
        Each row is (sequence_id, chunk_id).
    """
    if cu_seqlens_cpu is not None:
        # Calculate number of chunks for each sequence: ceil(seq_len / chunk_size)
        indices = torch.cat([torch.arange(n, device=cu_seqlens.device)
                            for n in cdiv_pt(torch.diff(cu_seqlens_cpu), chunk_size).tolist()])
        # Stack sequence_id and chunk_id
        # indices.eq(0) finds where chunk_id resets to 0 (start of new sequence)
        # cumsum counts these resets to get sequence_id
        return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

    indices = torch.cat([torch.arange(n) for n in cdiv_pt(torch.diff(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

def chunk_gated_delta_rule_fwd_kernel_varlen(
    k_ref,      # [B, T, 1, K_PADSIZE // 128, 128] -> [B, T, K_PADSIZE // 128, 128]
    v_ref,      # [B, T, 1, V_PADSIZE//BV, BV * 2]     -> [B, T, V_PADSIZE//BV, BV]
    w_ref,      # [B, T, 1, K_PADSIZE // 128, 128] -> [B, T, K_PADSIZE // 128, 128]
    g_ref,      # [B, T, H, 128]
    gk_ref,     # [B, T, H, K]
    h0_ref,     # [N, H, V, K]
    seqlens_ref,# [N + 1]
    chunk_offsets_ref, # [N + 1]

    # output
    h_ref,      # [B, NT, H, V, K]
    v_new_ref,  # [H, B, T, V_PADSIZE//BV, BV]
    ht_ref,     # [N, H, K, V]

    B,
    T,
    NT,
    H,
    K,
    V,
    BT,
    BV,
    USE_G,
    USE_GK,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
    SAVE_NEW_VALUE,
    USE_EXP2,
    IS_VARLEN = True,
):
  assert IS_VARLEN == True
  k_ref = k_ref.reshape(B, T, -1, 128)
  v_ref = v_ref.reshape(B, T, -1, 2, BV)
  w_ref = w_ref.reshape(B, T, -1, 128)

  idx_v, idx_nh = pl.program_id(0), pl.program_id(1)
  idx_n, idx_h = idx_nh // H, idx_nh % H

  if IS_VARLEN:
    bos = seqlens_ref[idx_n]
    eos = seqlens_ref[idx_n + 1]
    real_T = eos - bos
    real_NT = (real_T + BT - 1) // BT
    boh = chunk_offsets_ref[idx_n]
  else:
    bos = idx_n * T
    eos = bos + T
    real_NT = (T + BT - 1) // BT
    boh = idx_n * real_NT

  b_h1 = jnp.zeros([64, BV], dtype=jnp.float32)
  b_h2 = jnp.zeros([64, BV], dtype=jnp.float32)
  b_h3 = jnp.zeros([64, BV], dtype=jnp.float32)
  b_h4 = jnp.zeros([64, BV], dtype=jnp.float32)

  if USE_INITIAL_STATE:
    b_h1 += h0_ref[idx_n, idx_h, pl.ds(idx_v * BV, BV), 0:64].astype(jnp.float32).transpose(1, 0)
    if K > 64:
      b_h2 += h0_ref[idx_n, idx_h, pl.ds(idx_v * BV, BV), 64:128].astype(jnp.float32).transpose(1, 0)
    if K > 128:
      b_h3 += h0_ref[idx_n, idx_h, pl.ds(idx_v * BV, BV), 128:192].astype(jnp.float32).transpose(1, 0)
    if K > 192:
      b_h4 += h0_ref[idx_n, idx_h, pl.ds(idx_v * BV, BV), 192:256].astype(jnp.float32).transpose(1, 0)

  def loop_real_NT(idx_t, carry):
    b_h1, b_h2, b_h3, b_h4 = carry
    len_k1 = min(K, 64)
    h_ref[0, boh + idx_t, idx_h, pl.ds(idx_v * BV, BV), 0:len_k1] = b_h1.astype(h_ref.dtype).transpose(1, 0)[:, :len_k1]
    if K > 64:
      len_k2 = min(K, 128) - 64
      h_ref[0, boh + idx_t, idx_h, pl.ds(idx_v * BV, BV), 64:64+len_k2] = b_h2.astype(h_ref.dtype).transpose(1, 0)[:, :len_k2]
    if K > 128:
      len_k3 = min(K, 192) - 128
      h_ref[0, boh + idx_t, idx_h, pl.ds(idx_v * BV, BV), 128:128+len_k3] = b_h3.astype(h_ref.dtype).transpose(1, 0)[:, :len_k3]
    if K > 192:
      len_k4 = min(K, 256) - 192
      h_ref[0, boh + idx_t, idx_h, pl.ds(idx_v * BV, BV), 192:192+len_k4] = b_h4.astype(h_ref.dtype).transpose(1, 0)[:, :len_k4]


    # m_t = (idx_t * BT + jnp.arange(0, BT)) < real_T
    # m_t_2d = m_t.astype(jnp.int32)[:,None].astype(jnp.bool)

    valid_len = real_T - idx_t * BT
    mask = jnp.arange(BT)[:, None] < valid_len

    b_w = w_ref[0, pl.ds(bos + idx_t * BT, BT), 0, :][:, 0:64]
    b_w = jnp.where(mask, b_w, 0)

    b_v = jnp.dot(b_w.astype(jnp.float32), b_h1, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 64:
      b_w = w_ref[0, pl.ds(bos + idx_t * BT, BT), 0, :][:, 64:128]
      b_w = jnp.where(mask, b_w, 0)
      b_v += jnp.dot(b_w.astype(jnp.float32), b_h2, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 128:
      b_w = w_ref[0, pl.ds(bos + idx_t * BT, BT), 1, :][:, 0:64]
      b_w = jnp.where(mask, b_w, 0)
      b_v += jnp.dot(b_w.astype(jnp.float32), b_h3, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 192:
      b_w = w_ref[0, pl.ds(bos + idx_t * BT, BT), 1, :][:, 64:128]
      b_w = jnp.where(mask, b_w, 0)
      b_v += jnp.dot(b_w.astype(jnp.float32), b_h4, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    b_v_raw = v_ref[0, pl.ds(bos + idx_t * BT, BT), idx_v, 0, :].astype(b_v.dtype)
    b_v_raw = jnp.where(mask, b_v_raw, 0)
    b_v = b_v_raw - b_v

    if SAVE_NEW_VALUE:
      v_new_slice = v_new_ref[idx_h, 0, pl.ds(bos + idx_t * BT, BT), idx_v, 0:BV]
      v_new_val = b_v.astype(v_new_ref.dtype)
      v_new_ref[idx_h, 0, pl.ds(bos + idx_t * BT, BT), idx_v, 0:BV] = jnp.where(mask, v_new_val, v_new_slice)

    last_idx = jnp.minimum((idx_t + 1) * BT, real_T) - 1

    if USE_G:
      m_t = (idx_t * BT + jnp.arange(0, BT)) < real_T
      b_g_last = g_ref[0, bos + last_idx, idx_h, 0]
      b_g = g_ref[0, pl.ds(bos + idx_t * BT, BT), idx_h, :]
      # Masking g is good practice though m_t handles the exp logic
      # b_g = jnp.where(mask, b_g, 0)
      b_g = b_g[:BT, :1].reshape(BT)
      if USE_EXP2:
        b_v = b_v * jnp.where(m_t, jnp.exp2(b_g_last - b_g), 0)[:, None]
        b_g_last = jnp.exp2(b_g_last)
      else:
        b_v = b_v * jnp.where(m_t, jnp.exp(b_g_last - b_g), 0)[:, None]
        b_g_last = jnp.exp(b_g_last)
      b_h1 *= b_g_last
      if K > 64:
        b_h2 *= b_g_last
      if K > 128:
        b_h3 *= b_g_last
      if K > 192:
        b_h4 *= b_g_last

    if USE_GK:
      o_k1 = jnp.arange(0, 64)
      b_gk_last1 = jnp.where(o_k1 < K,
                      gk_ref[0, bos + last_idx, idx_h, 0, :],
                      0
                    ).astype(jnp.float32)
      if USE_EXP2:
        b_h1 *= jnp.exp2(b_gk_last1)[:, None]
      else:
        b_h1 *= jnp.exp(b_gk_last1)[:, None]

      if K > 64:
        o_k2 = 64 + o_k1
        b_gk_last2 = jnp.where(o_k2 < K, gk_ref[0, bos + last_idx, idx_h, 1, :], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h2 *= jnp.exp2(b_gk_last2)[:, None]
        else:
          b_h2 *= jnp.exp(b_gk_last2)[:, None]

      if K > 128:
        o_k3 = 128 + o_k1
        b_gk_last3 = jnp.where(o_k3 < K, gk_ref[0, bos + last_idx, idx_h, 2, :], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h3 *= jnp.exp2(b_gk_last3)[:, None]
        else:
          b_h3 *= jnp.exp(b_gk_last3)[:, None]

      if K > 192:
        o_k4 = 192 + o_k1
        b_gk_last4 = jnp.where(o_k4 < K, gk_ref[0, bos + last_idx, idx_h, 3, :], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h4 *= jnp.exp2(b_gk_last4)[:, None]
        else:
          b_h4 *= jnp.exp(b_gk_last4)[:, None]

    # b_v = b_v.astype(k_ref.dtype)

    b_k = k_ref[0, pl.ds(bos + idx_t * BT, BT), 0, :][:, 0:64]
    b_k = jnp.where(mask, b_k, 0)
    b_h1 += jnp.dot(b_k.astype(jnp.float32).T, b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 64:
      b_k = k_ref[0, pl.ds(bos + idx_t * BT, BT), 0, :][:, 64:128]
      b_k = jnp.where(mask, b_k, 0)
      b_h2 += jnp.dot(b_k.astype(jnp.float32).T, b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    if K > 128:
      b_k = k_ref[0, pl.ds(bos + idx_t * BT, BT), 1, :][:, 0:64]
      b_k = jnp.where(mask, b_k, 0)
      b_h3 += jnp.dot(b_k.astype(jnp.float32).T, b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 192:
      b_k = k_ref[0, pl.ds(bos + idx_t * BT, BT), 1, :][:, 64:128]
      b_k = jnp.where(mask, b_k, 0)
      b_h4 += jnp.dot(b_k.astype(jnp.float32).T, b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    return b_h1, b_h2, b_h3, b_h4

  carry = (b_h1, b_h2, b_h3, b_h4)
  carry = jax.lax.fori_loop(0, real_NT, loop_real_NT, carry)
  b_h1, b_h2, b_h3, b_h4 = carry

  if STORE_FINAL_STATE:
    len_k1 = min(K, 64)
    ht_ref[idx_n, idx_h, 0:len_k1, :] = b_h1.astype(ht_ref.dtype)[:len_k1, :]
    if K > 64:
      len_k2 = min(K, 128) - 64
      ht_ref[idx_n, idx_h, 64:64+len_k2, :] = b_h2.astype(ht_ref.dtype)[:len_k2, :]
    if K > 128:
      len_k3 = min(K, 192) - 128
      ht_ref[idx_n, idx_h, 128:128+len_k3, :] = b_h3.astype(ht_ref.dtype)[:len_k3, :]
    if K > 192:
      len_k4 = min(K, 256) - 192
      ht_ref[idx_n, idx_h, 192:192+len_k4, :] = b_h4.astype(ht_ref.dtype)[:len_k4, :]

def chunk_gated_delta_rule_fwd_h_varlen(
    k: jax.Array,
    w: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    gk: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    BV: int = 64,
    save_new_value: bool = True,
    seqlens: jax.Array | None = None,
    chunk_indices: jax.Array | None = None,
    use_exp2: bool = False,
):

  B, T, H, K, V = *k.shape, v.shape[-1]
  BT = chunk_size
  K_BPE = k.dtype.itemsize
  W_BPE = w.dtype.itemsize
  V_BPE = v.dtype.itemsize
  K_PADSIZE = int(AlignUP(K, 512 / K_BPE))
  V_PADSIZE = int(AlignUP(V, 512 / V_BPE))

  assert ((seqlens == None) or (seqlens != None and chunk_indices != None))
  assert K <= 256, "current kernel does not support head dimension larger than 256."
  assert k.shape == (B, T, H, K)
  assert w.shape == (B, T, H, K)
  assert v.shape == (B, T, H, V)
  assert ((seqlens == None) or (seqlens != None and B == 1))
  assert ((g is None) or (g.shape == (B, T, H)))
  assert ((gk is None) or (gk.shape == (B, T, H, K)))
  if seqlens is None:
    N, NT, chunk_offsets = B, math.ceil(T / BT), None
  else:
    N, NT, chunk_offsets = len(seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(seqlens, BT)
  assert ((initial_state is None) or (initial_state.shape == (N, H, K, V)))

  if initial_state is not None:
    initial_state = initial_state.transpose(0, 1, 3, 2)
    # [N, H, K, V] -> [N, H, V, K]
    initial_state = pad_to_multiple(initial_state, 512 // K_BPE, -1, 0)


  # [B, T, H, K] -> [B, T, H, K_PADSIZE]
  # -> [B, T, H, K_PADSIZE // 128, 128]
  k_paded = k
  k_paded = pad_to_multiple(k_paded, 512 // K_BPE, -1, 0)
  k_paded = k_paded.reshape(B, T, H, -1, 128)

  # [B, T, H, K] -> [B, T, H, K_PADSIZE]
  # -> [B, T, H, K_PADSIZE // 128, 128]
  w_paded = w
  w_paded = pad_to_multiple(w_paded, 512 // W_BPE, -1, 0)
  w_paded = w_paded.reshape(B, T, H, -1, 128)

  # [B, T, H, V] -> [B, T, H, V_PADSIZE]
  # -> [B, T, H, V_PADSIZE//BV, BV]
  # -> [B, T, H, V_PADSIZE//BV, BV * 2]
  v_paded = v
  v_paded = pad_to_multiple(v_paded, BV, -1, 0)
  v_paded = v_paded.reshape(B, T, H, -1, BV)
  v_paded = pad_to_multiple(v_paded, BV*2, -1, 0)

  h_shape = [B, NT, H, K, V]
  v_new_shape = [B, T, H, V]
  final_state_shape = [N, H, K, V]
  h = jnp.zeros(h_shape, dtype=k.dtype)
  v_new = jnp.zeros(v_new_shape, dtype=jnp.float32)
  final_state = jnp.zeros(final_state_shape, dtype=jnp.float32)

  # [B, NT, H, K, V] -> [B, NT, H, V, K]
  h = h.transpose(0, 1, 2, 4, 3)

  # [B, T, H, V] -> [H, B, T, V] -> [H, B, T, V_PADSIZE]
  # -> [H, B, T, V_PADSIZE//BV, BV]
  v_new = v_new.transpose(2, 0, 1, 3)
  v_new = pad_to_multiple(v_new, BV, -1, 0)
  v_new = v_new.reshape(H, B, T, -1, BV)

  if g is not None:
    g_fp32 = g.astype(jnp.float32)
    g_fp32 = g_fp32.reshape(B, T, H, 1)
    g_fp32 = pad_to_multiple(g_fp32, 128, -1, 0)
  else:
    g_fp32 = None

  if gk is not None:
    gk_fp32 = gk.astype(jnp.float32)
    gk_fp32 = pad_to_multiple(gk_fp32, 128, -1, 0)
    gk_fp32 = gk_fp32.reshape(B, T, H, -1, 64)
  else:
    gk_fp32 = jnp.zeros([B, T, H, K], jnp.float32)

  h_spec = jax.ShapeDtypeStruct([B, NT, H, V, K], h.dtype)
  v_new_spec = jax.ShapeDtypeStruct([H, B, T, V_PADSIZE//BV, BV], jnp.float32)
  final_state_spec = jax.ShapeDtypeStruct(final_state_shape, jnp.float32)

  k_blockspec = pl.BlockSpec([B, T, 1, K_PADSIZE//128, 128], index_map = lambda v, bh: (0, 0, bh % H, 0, 0))
  w_blockspec = pl.BlockSpec([B, T, 1, K_PADSIZE//128, 128], index_map = lambda v, bh: (0, 0, bh % H, 0, 0))
  v_blockspec = pl.BlockSpec([B, T, 1, v_paded.shape[3], BV * 2], index_map = lambda v, bh: (0, 0, bh % H, 0, 0))
  g_blockspec = pl.BlockSpec([B, T, H, 128], index_map = lambda v, bh: (0, 0, 0, 0))
  gk_blockspec = pl.BlockSpec([B, T, H, gk_fp32.shape[-2], 64], index_map = lambda v, bh: (0, 0, 0, 0, 0))
  init_blockspec = pl.BlockSpec([N, H, V, K_PADSIZE], index_map = lambda v, bh: (0, 0, 0, 0))
  seqlens_blockspec = pl.BlockSpec([N + 1], index_map = lambda v, bh: (0,), memory_space = pltpu.MemorySpace.SMEM)
  chunk_offsets_blockspec = pl.BlockSpec([N + 1], index_map = lambda v, bh: (0,), memory_space = pltpu.MemorySpace.SMEM)

  h_blockspec = pl.BlockSpec([B, NT, H, V, K], lambda v, bh : (0, 0, 0, 0, 0))
  v_new_blockspec = pl.BlockSpec([H, B, T, V_PADSIZE//BV, BV], lambda v, bh : (0, 0, 0, 0, 0))
  final_out_blockspec = pl.BlockSpec([N, H, K, V], lambda v, bh : (0, 0, 0, 0))

  grid = (math.ceil(V / BV), N * H)
  h, v_out, final_out = pl.pallas_call(
    functools.partial(
        chunk_gated_delta_rule_fwd_kernel_varlen,
        B=B,
        T=T,
        NT=NT,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        USE_G=(g is not None),
        USE_GK=(gk is not None),
        USE_INITIAL_STATE=(initial_state is not None),
        STORE_FINAL_STATE=(final_state is not None),
        SAVE_NEW_VALUE=(v_new is not None),
        USE_EXP2=use_exp2,
        IS_VARLEN=(seqlens is not None),
    ),
    grid=grid,
    in_specs=[k_blockspec, v_blockspec, w_blockspec,
              g_blockspec if (g is not None) else None,
              gk_blockspec if (gk is not None) else None,
              init_blockspec if (initial_state is not None) else None,
              seqlens_blockspec, chunk_offsets_blockspec],
    out_shape=[h_spec, v_new_spec, final_state_spec],
    out_specs=[h_blockspec, v_new_blockspec, final_out_blockspec],
  )(k_paded, v_paded, w_paded, g_fp32, gk_fp32, initial_state, seqlens, chunk_offsets)

  if save_new_value:
    v_out = v_out.reshape(H, B, T, V_PADSIZE).transpose(1, 2, 0, 3)
    v_out = v_out[:,:,:,:V]
  h = h.transpose(0, 1, 2, 4, 3)
  return h, (v_out if save_new_value else None), (final_out if output_final_state else None)

def chunk_gla_fwd_o_gk_kernel_varlen(
    q_ref, v_ref, g_ref, h_ref, A_ref, chunk_indices_ref, cu_seqlens_ref, # Inputs
    o_ref,                                                                # Output
    scale,
    TotalT, TotalChunks, H, K, V, BT, BV,
    USE_EXP2
):
    idx_v, idx_chunk_h = pl.program_id(0), pl.program_id(1)
    idx_chunk = idx_chunk_h // H

    seq_id = chunk_indices_ref[idx_chunk, 0]
    local_chunk_id = chunk_indices_ref[idx_chunk, 1]

    bos = cu_seqlens_ref[seq_id]
    eos = cu_seqlens_ref[seq_id + 1]

    chunk_start = bos + local_chunk_id * BT
    valid_len = jnp.minimum(BT, eos - chunk_start)
    mask_t = jnp.arange(BT)[:, None] < valid_len

    # q_ref is [1, 1, 1, BT, 128]
    b_q = q_ref[0, 0, 0, :, :]
    b_g = g_ref[0, 0, 0, :, :]

    # A_ref is [1, 1, 1, BT, 128] -> Slice to [BT, BT]
    b_A_full = A_ref[0, 0, 0, :, :]
    b_A = b_A_full[:, :BT]

    # v_ref is [1, 1, 1, BT, 128]
    # BV=128
    b_v = v_ref[0, 0, 0, :, pl.ds(idx_v*BV, BV)]

    # h_ref: [1, 1, 128, 128] (Sliced H)
    b_h = h_ref[0, 0, :, pl.ds(idx_v*BV, BV)]

    # Masking
    mask_broad = mask_t[None, ...]

    # Apply mask to q, g
    b_q = jnp.where(mask_t, b_q, 0)
    # b_g = jnp.where(mask_t, b_g, 0)

    if USE_EXP2:
        b_qg = b_q.astype(jnp.float32) * jnp.exp2(b_g.astype(jnp.float32))
    else:
        b_qg = b_q.astype(jnp.float32) * jnp.exp(b_g.astype(jnp.float32))

    # Inter-chunk: (q * exp(g)) @ h
    # [BT, 128] @ [128, BV] -> [BT, BV]
    b_inter = jnp.matmul(b_qg, b_h.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    b_inter *= scale

    # Intra-chunk: A @ v
    # Mask A
    mask_causal = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]
    mask_intra = mask_t & mask_causal # [BT, BT]

    b_A = jnp.where(mask_intra, b_A, 0)
    b_v = jnp.where(mask_t, b_v, 0)

    # [BT, BT] @ [BT, BV] -> [BT, BV]
    b_intra = jnp.matmul(b_A.astype(jnp.float32), b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    b_o = b_inter + b_intra # [BT, BV]

    # Store
    # o_ref: [1, 1, 1, BT, 128]
    o_ref[0, 0, 0, :, pl.ds(idx_v*BV, BV)] = b_o.astype(o_ref.dtype)

def chunk_gla_fwd_o_gk_varlen(
    q: jax.Array,
    v: jax.Array,
    g: jax.Array,
    A: jax.Array,
    h: jax.Array,
    chunk_indices: jax.Array,
    seqlens: jax.Array,
    scale: float,
    chunk_size: int = 64,
    use_exp2: bool = False,
):
    TotalT, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    TotalChunks = chunk_indices.shape[0]

    assert q.shape == (TotalT, H, K)
    assert v.shape == (TotalT, H, V)
    assert g.shape == (TotalT, H, K)
    assert A.shape == (TotalT, H, BT)
    # h: [TotalChunks, H, K, V]

    # Save original shapes for slicing back
    V_orig = V

    # Reshape to [1, TotalT, H, K] for Pallas consistency
    q = q[None, ...]
    v = v[None, ...]
    g = g[None, ...]
    A = A[None, ...]

    # Pad TotalT to multiple of BT (chunk_size)
    q = pad_to_multiple(q, BT, 1, 0)
    v = pad_to_multiple(v, BT, 1, 0)
    g = pad_to_multiple(g, BT, 1, 0)
    A = pad_to_multiple(A, BT, 1, 0)

    PaddedT = q.shape[1]

    # Pad K, V, A_last to 128
    q = pad_to_multiple(q, 128, 3, 0)
    g = pad_to_multiple(g, 128, 3, 0)
    v = pad_to_multiple(v, 128, 3, 0)
    A = pad_to_multiple(A, 128, 3, 0) # Pad BT dim to 128

    # Pad h
    # h: [TotalChunks, H, K, V]
    h = pad_to_multiple(h, 128, 2, 0) # Pad K
    h = pad_to_multiple(h, 128, 3, 0) # Pad V

    # Transpose inputs to [H, 1, T, K] to satisfy Pallas/Mosaic constraints
    # (Second to last dim must be divisible by 8)
    q = q.transpose(2, 0, 1, 3) # [H, 1, T, K]
    v = v.transpose(2, 0, 1, 3)
    g = g.transpose(2, 0, 1, 3)
    A = A.transpose(2, 0, 1, 3)

    K_padded = q.shape[3]
    V_padded = v.shape[3]
    BV = 128 # Block size for V (Must be 128 for alignment)

    # Reshape T -> [TotalChunks, BT]
    # This assumes PaddedT == TotalChunks * BT, which is true if seqlens are padded/aligned.
    assert PaddedT % BT == 0
    NumChunks = PaddedT // BT

    # [H, 1, NumChunks, BT, 128]
    q = q.reshape(H, 1, NumChunks, BT, 128)
    v = v.reshape(H, 1, NumChunks, BT, 128)
    g = g.reshape(H, 1, NumChunks, BT, 128)
    A = A.reshape(H, 1, NumChunks, BT, 128)

    o_shape = v.shape # [H, 1, NumChunks, BT, 128]
    o = jnp.zeros(o_shape, dtype=v.dtype)

    o_spec = jax.ShapeDtypeStruct(o_shape, o.dtype)

    # BlockSpecs
    # Slice H (dim 0) and Chunk (dim 2)
    # Index Map: (ch % H, 0, ch // H, 0, 0)
    # Kernel sees: [1, 1, 1, 64, 128]

    slice_shape_q = (1, 1, 1, BT, K_padded)
    slice_shape_v = (1, 1, 1, BT, V_padded)
    slice_shape_g = (1, 1, 1, BT, K_padded)
    slice_shape_A = (1, 1, 1, BT, 128)

    q_blockspec = pl.BlockSpec(slice_shape_q, index_map=lambda v, ch: (ch % H, 0, ch // H, 0, 0))
    v_blockspec = pl.BlockSpec(slice_shape_v, index_map=lambda v, ch: (ch % H, 0, ch // H, 0, 0))
    g_blockspec = pl.BlockSpec(slice_shape_g, index_map=lambda v, ch: (ch % H, 0, ch // H, 0, 0))
    A_blockspec = pl.BlockSpec(slice_shape_A, index_map=lambda v, ch: (ch % H, 0, ch // H, 0, 0))

    # h: [TotalChunks, H, K, V]
    # Slice H (dim 1) and Chunk (dim 0)
    # Index Map: (ch // H, ch % H, 0, 0)
    # Kernel sees: [1, 1, 128, 128]
    # h_shape: [TotalChunks, H, 128, 128]
    slice_shape_h = (1, 1, 128, 128)
    h_blockspec = pl.BlockSpec(slice_shape_h, index_map=lambda v, ch: (ch // H, ch % H, 0, 0))

    ci_blockspec = pl.BlockSpec([TotalChunks, 2], index_map=lambda v, ch: (0, 0), memory_space=pltpu.MemorySpace.SMEM)
    sl_blockspec = pl.BlockSpec([seqlens.shape[0]], index_map=lambda v, ch: (0,), memory_space=pltpu.MemorySpace.SMEM)

    o_blockspec = pl.BlockSpec(slice_shape_v, index_map=lambda v, ch: (ch % H, 0, ch // H, 0, 0))

    # Grid: (V_blocks, TotalChunks * H)
    grid = (int(V_padded // BV), TotalChunks * H)

    o = pl.pallas_call(
        functools.partial(
            chunk_gla_fwd_o_gk_kernel_varlen,
            scale=scale,
            TotalT=PaddedT,
            TotalChunks=TotalChunks,
            H=H, K=K_padded, V=V_padded, BT=BT, BV=BV,
            USE_EXP2=use_exp2
        ),
        grid=grid,
        in_specs=[q_blockspec, v_blockspec, g_blockspec, h_blockspec, A_blockspec, ci_blockspec, sl_blockspec],
        out_shape=o_spec,
        out_specs=o_blockspec
    )(q, v, g, h, A, chunk_indices, seqlens)

    # o is [H, 1, NumChunks, BT, 128]
    # Reshape to [H, 1, PaddedT, 128]
    o = o.reshape(H, 1, PaddedT, 128)

    # Transpose back to [1, T, H, 128]
    o = o.transpose(1, 2, 0, 3)

    # Remove B=1 and slice
    o = o[0] # [T, H, 128]
    o = o[:TotalT, :, :V_orig]

    return o