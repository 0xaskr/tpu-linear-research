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
    为变长序列生成分块索引。

    Returns:
        torch.LongTensor: A tensor of shape [Num_Total_Chunks, 2].
        Each row is (sequence_id, chunk_id).
        每一行包含两个标记：(句子ID, 该句内的块ID)。
    """
    if cu_seqlens_cpu is not None:
        # Calculate number of chunks for each sequence: ceil(seq_len / chunk_size)
        # 计算每个句子被分成了多少个块
        indices = torch.cat([torch.arange(n, device=cu_seqlens.device)
                            for n in cdiv_pt(torch.diff(cu_seqlens_cpu), chunk_size).tolist()])
        # Stack sequence_id and chunk_id
        # indices.eq(0) finds where chunk_id resets to 0 (start of new sequence)
        # cumsum counts these resets to get sequence_id
        return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

    # prepare_lens = torch.diff
    indices = torch.cat([torch.arange(n) for n in cdiv_pt(torch.diff(cu_seqlens), chunk_size).tolist()])
    # 这个函数只生成逻辑上的 (Seq_ID, Chunk_ID) 对，不涉及实际数据的读取。
    # 越界保护机制 (Boundary Check):
    # 下游的 Triton Kernel 在使用这些 ID 时，必须执行以下逻辑：
    # 1. 计算当前 Chunk 的起始 Token： start_token_idx = cu_seqlens[seq_id] + chunk_id * chunk_size
    # 2. 计算当前 Chunk 的有效长度： visible_len = min(chunk_size, seq_len - chunk_id * chunk_size)
    # 3. 使用 Mask 加载数据： tl.load(..., mask=offsets < visible_len, other=0)
    # 因此，即使最后一个 Chunk 不满 (Partial Chunk)，Kernel 也会通过 Mask 防止越界。
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

# TODO(0xaskr) need pad
def chunk_gated_delta_rule_fwd_kernel_varlen(
    k_ref,      # [H, B, T, K_PADED // 128, 128]
    v_ref,      # [H, B, T, V // 64, 2, 256]
    w_ref,      # [H, B, T, K_PADED // 128, 128]
    g_ref,      # [B, T, H]
    gk_ref,     # [B, T, H, K]
    h0_ref,     # [N, H, V, K]
    seqlens_ref,# [N + 1]
    chunk_offsets_ref, # [N + 1]

    # output
    h_ref,      # [N, NT, H, V, K]
    v_new_ref,  # [H, B, T, V// 64, 2, 256]
    ht_ref,     # [1, H, K, BV]

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

  # if SAVE_NEW_VALUE:
  #   v_new + =

  if USE_INITIAL_STATE:
    h0_arr = h0_ref[idx_n, idx_h, pl.ds(idx_v * BV, BV), :].astype(jnp.float32)
    h0_arr = h0_arr[:, 0:64].transpose(1, 0)
    # b_h1 += h0_ref[idx_n, idx_h, 0:64, pl.ds(idx_v * BV, BV)].astype(jnp.float32)
    b_h1 += h0_arr
    # if K > 64:
    #   b_h2 += h0_ref[idx_n, idx_h, 64:128, pl.ds(idx_v * BV, BV)].astype(jnp.float32)
    # if K > 128:
    #   b_h3 += h0_ref[idx_n, idx_h, 128:192, pl.ds(idx_v * BV, BV)].astype(jnp.float32)
    # if K > 192:
    #   b_h4 += h0_ref[idx_n, idx_h, 192:256, pl.ds(idx_v * BV, BV)].astype(jnp.float32)

  def loop_real_NT(idx_t, carry):
    b_h1, b_h2, b_h3, b_h4 = carry
    # h_ref[0, boh + idx_t, idx_h, 0:64, pl.ds(idx_v * BV, BV)] = b_h1.astype(h_ref.dtype)
    h_ref[0, boh + idx_t, idx_h, pl.ds(idx_v * BV, BV), 0:64] = b_h1.astype(h_ref.dtype).transpose(1, 0)
    # if K > 64:
    #   h_ref[0, boh + idx_t, idx_h, 64:128, pl.ds(idx_v * BV, BV)] = b_h2.astype(h_ref.dtype)
    # if K > 128:
    #   h_ref[0, boh + idx_t, idx_h, 128:192, pl.ds(idx_v * BV, BV)] = b_h3.astype(h_ref.dtype)
    # if K > 192:
    #   h_ref[0, boh + idx_t, idx_h, 192:256, pl.ds(idx_v * BV, BV)] = b_h4.astype(h_ref.dtype)


    m_t = (idx_t * BT + jnp.arange(0, BT)) < real_T
    m_t_2d = m_t.astype(jnp.int32)[:,None].astype(jnp.bool)

    # b_w = w_ref[0, pl.ds(bos + idx_t * BT, BT), idx_h, 0:64]
    w_tmp = w_ref[idx_h, 0, pl.ds(bos + idx_t * BT, BT), 0, :]
    b_w = w_tmp[:, 0:64]
    b_w = jnp.where(m_t_2d, b_w, 0)
    b_v = jnp.dot(b_w.astype(jnp.float32), b_h1, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    # if K > 64:
    #   b_w2 = w_ref[0, pl.ds(bos + idx_t * BT, BT), idx_h, 64:128]
    #   b_w2 = jnp.where(m_t[:, None], b_w2, 0)
    #   b_v += jnp.dot(b_w2.astype(jnp.float32), b_h2, preferred_element_type=jnp.float32)
    # if K > 128:
    #   b_w3 = w_ref[0, pl.ds(bos + idx_t * BT, BT), idx_h, 128:192]
    #   b_w3 = jnp.where(m_t[:, None], b_w3, 0)
    #   b_v += jnp.dot(b_w3.astype(jnp.float32), b_h3, preferred_element_type=jnp.float32)
    # if K > 192:
    #   b_w4 = w_ref[0, pl.ds(bos + idx_t * BT, BT), idx_h, 192:256]
    #   b_w4 = jnp.where(m_t[:, None], b_w4, 0)
    #   b_v += jnp.dot(b_w4.astype(jnp.float32), b_h4, preferred_element_type=jnp.float32)

    b_v_raw = v_ref[idx_h, 0, pl.ds(bos + idx_t * BT, BT), pl.ds(idx_v, 1), 0, :].astype(b_v.dtype)
    b_v_raw = b_v_raw.reshape(BT, 256)[:, 0:BV]
    print("b_v_raw.shape = ", b_v_raw.shape)
    b_v_raw = jnp.where(m_t_2d, b_v_raw, 0)
    b_v = b_v_raw - b_v

    if SAVE_NEW_VALUE:
      v_new_ref[idx_h, 0, pl.ds(bos + idx_t * BT, BT), idx_v, :, :] = pad_to_multiple(b_v, 512, -1, 0).reshape(BT, 2, 256).astype(v_new_ref.dtype)

    last_idx = jnp.minimum((idx_t + 1) * BT, real_T) - 1

    if USE_G:
      m_t = (idx_t * BT + jnp.arange(0, BT)) < real_T
      b_g_last = g_ref[0, bos + last_idx, idx_h].astype(jnp.float32)
      # g_trans = jnp.transpose(g_ref[...], [0, 2, 1])  # B, T, H -> B, H, T
      # b_g = g_trans[0, idx_h, pl.ds(idx_t * BT, BT)]
      b_g = g_ref[0, pl.ds(bos + idx_t * BT, BT), idx_h]
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
                      gk_ref[0, bos + last_idx, idx_h, 0:64],
                      0
                    ).astype(jnp.float32)
      if USE_EXP2:
        b_h1 *= jnp.exp2(b_gk_last1)[:, None]
      else:
        b_h1 *= jnp.exp(b_gk_last1)[:, None]

      if K > 64:
        o_k2 = 64 + o_k1
        b_gk_last2 = jnp.where(o_k2 < K, gk_ref[0, bos + last_idx, idx_h, 64:128], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h2 *= jnp.exp2(b_gk_last2)[:, None]
        else:
          b_h2 *= jnp.exp(b_gk_last2)[:, None]

      if K > 128:
        o_k3 = 128 + o_k1
        b_gk_last3 = jnp.where(o_k3 < K, gk_ref[0, bos + last_idx, idx_h, 128:192], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h3 *= jnp.exp2(b_gk_last3)[:, None]
        else:
          b_h3 *= jnp.exp(b_gk_last3)[:, None]

      if K > 192:
        o_k4 = 192 + o_k1
        b_gk_last4 = jnp.where(o_k4 < K, gk_ref[0, bos + last_idx, idx_h, 192:256], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h4 *= jnp.exp2(b_gk_last4)[:, None]
        else:
          b_h4 *= jnp.exp(b_gk_last4)[:, None]

    # b_v = b_v.astype(k_ref.dtype)

    b_k = k_ref[0, pl.ds(bos + idx_t * BT, BT), idx_h, 0:64]
    b_k = k_ref[idx_h, 0, pl.ds(bos + idx_t * BT, BT), 0, :]
    b_k = b_k[:, 0:64]
    b_k = jnp.where(m_t_2d, b_k, 0).reshape(BT, 64).transpose(1, 0)
    b_h1 += jnp.dot(b_k.astype(jnp.float32), b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    # if K > 64:
    #   b_k2 = k_ref[0, pl.ds(bos + idx_t * BT, BT), idx_h, 64:128]
    #   b_k2 = jnp.where(m_t[:, None], b_k2, 0).reshape(BT, 64).transpose(1, 0)
    #   b_h2 += jnp.dot(b_k2.astype(jnp.float32), b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    # if K > 128:
    #   b_k3 = k_ref[0, pl.ds(bos + idx_t * BT, BT), idx_h, 128:192]
    #   b_k3 = jnp.where(m_t[:, None], b_k3, 0).reshape(BT, 64).transpose(1, 0)
    #   b_h3 += jnp.dot(b_k3.astype(jnp.float32), b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    # if K > 192:
    #   b_k4 = k_ref[0, pl.ds(bos + idx_t * BT, BT), idx_h, 192:256]
    #   b_k4 = jnp.where(m_t[:, None], b_k4, 0).reshape(BT, 64).transpose(1, 0)
    #   b_h4 += jnp.dot(b_k4.astype(jnp.float32), b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    return b_h1, b_h2, b_h3, b_h4

  carry = (b_h1, b_h2, b_h3, b_h4)
  carry = jax.lax.fori_loop(0, real_NT, loop_real_NT, carry)
  b_h1, b_h2, b_h3, b_h4 = carry

  if STORE_FINAL_STATE:
    ht_ref[idx_n, idx_h, 0:64, :] = b_h1.astype(ht_ref.dtype)
    if K > 64:
      ht_ref[idx_n, idx_h, 64:128, :] = b_h2.astype(ht_ref.dtype)
    if K > 128:
      ht_ref[idx_n, idx_h, 128:192, :] = b_h3.astype(ht_ref.dtype)
    if K > 192:
      ht_ref[idx_n, idx_h, 192:256, :] = b_h4.astype(ht_ref.dtype)

def chunk_gated_delta_rule_fwd_h_varlen(
    k: jax.Array,
    w: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    gk: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    seqlens: jax.Array | None = None,
    chunk_indices: jax.Array | None = None,
    use_exp2: bool = False,
):

  B, T, H, K, V = *k.shape, v.shape[-1]
  BT = chunk_size
  BV = 64   # auto tune
  K_BPE = k.dtype.itemsize
  W_BPE = w.dtype.itemsize
  V_BPE = v.dtype.itemsize
  K_PADSIZE = int(AlignUP(K, 512 / K_BPE))
  # V_PADSIZE = int(AlignUP(V, 512 / V_BPE))

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


  # [B, T, H, K] -> [H, B, T, K_PADSIZE] -> [H, B, T, K_PADSIZE // 128, 128]
  k_paded = pad_to_multiple(k, 512 // K_BPE, -1, 0)
  k_paded = k_paded.transpose(2, 0, 1, 3)
  k_paded = k_paded.reshape(H, B, T, -1, 128)

  # [B, T, H, K] -> [H, B, T, K_PADSIZE] -> [H, B, T, K_PADSIZE // 128, 128]
  w_paded = pad_to_multiple(w, 512 // W_BPE, -1, 0)
  w_paded = w_paded.transpose(2, 0, 1, 3)
  w_paded = w_paded.reshape(H, B, T, -1, 128)

  # [B, T, H, V] -> [H, B, T, V] -> [B, T, H, V_PADSIZE]
  # -> [B, T, H, V_PADSIZE//BV, BV]
  v_paded = v.transpose(2, 0, 1, 3)
  v_paded = pad_to_multiple(v_paded, BV, -1, 0)
  v_paded = v_paded.reshape(B, T, H, -1, BV)

  h_shape = [B, NT, H, K, V]
  v_new_shape = [B, T, H, V]
  final_state_shape = [N, H, K, V]
  h = jnp.zeros(h_shape, dtype=k.dtype)
  v_new = jnp.zeros(v_new_shape, dtype=v.dtype) if save_new_value else None
  final_state = jnp.zeros(final_state_shape, dtype=jnp.float32) if output_final_state else None

  if g is not None:
    g_fp32 = g.astype(jnp.float32)
  else:
    g_fp32 = None

  if gk is not None:
    gk_fp32 = gk.astype(jnp.float32)
  else:
    gk_fp32 = None

  h_spec = jax.ShapeDtypeStruct(h_shape, h.dtype)
  v_new_spec = jax.ShapeDtypeStruct(v_new_shape, v.dtype)
  final_state_spec = jax.ShapeDtypeStruct(final_state_shape, jnp.float32)

  k_blockspec = pl.BlockSpec([H, B, T, K_PADSIZE//128, 128], index_map = lambda v, bh: (0, 0, 0, 0, 0))
  w_blockspec = pl.BlockSpec([H, B, T, K_PADSIZE//128, 128], index_map = lambda v, bh: (0, 0, 0, 0, 0))
  v_blockspec = pl.BlockSpec([B, T, H, v_paded.shape[3], BV], index_map = lambda v, bh: (0, 0, 0, 0, 0))
  g_blockspec = pl.BlockSpec([B, T, H], index_map = lambda v, bh: (0, 0, 0))
  gk_blockspec = pl.BlockSpec([B, T, H, K], index_map = lambda v, bh: (0, 0, 0, 0))
  init_blockspec = pl.BlockSpec([N, H, K, V], index_map = lambda v, bh: (0, 0, 0, 0))
  seqlens_blockspec = pl.BlockSpec([N + 1], index_map = lambda v, bh: (0,), memory_space = pltpu.MemorySpace.SMEM)
  chunk_offsets_blockspec = pl.BlockSpec([N + 1], index_map = lambda v, bh: (0,), memory_space = pltpu.MemorySpace.SMEM)

  h_blockspec = pl.BlockSpec([B, NT, H, K, V], lambda v, bh : (0, 0, 0, 0, 0))
  v_new_blockspec = pl.BlockSpec([B, T, H, V], lambda v, bh : (0, 0, 0, 0))
  final_out_blockspec = pl.BlockSpec([N, H, K, V], lambda v, bh : (0, 0, 0, 0))

  grid = (math.ceil(V / BV), N * H)
  h, v_out, final_out = pl.pallas_call(
    functools.partial(
        chunk_gated_delta_rule_fwd_kernel_varlen,
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
              init_blockspec if initial_state is not None else None,
              seqlens_blockspec, chunk_offsets_blockspec],
    out_shape=[h_spec, v_new_spec, final_state_spec],
    out_specs=[h_blockspec, v_new_blockspec, final_out_blockspec],
  )(k_paded, v_paded, w_paded, g_fp32, gk_fp32, initial_state, seqlens, chunk_offsets)

  return h, (v_out if save_new_value else None), (final_out if output_final_state else None)

def test_varlen():
    print("\nStarting Varlen Unit Test...")

    seqlens_list = [64, 128] # Simple lengths
    TotalT = sum(seqlens_list)
    B, H, K, V = 1, 4, 64, 64
    chunk_size = 64
    N = len(seqlens_list)

    rng_dtype = torch.bfloat16
    triton_dtype = torch.float32
    pallas_dtype = jnp.bfloat16

    torch.manual_seed(42)
    k = torch.randn((B, TotalT, H, K), dtype=rng_dtype)
    w = torch.randn((B, TotalT, H, K), dtype=rng_dtype)
    u = torch.randn((B, TotalT, H, V), dtype=rng_dtype)
    g = torch.randn((B, TotalT, H), dtype=rng_dtype)
    gk = torch.randn((B, TotalT, H, K), dtype=rng_dtype)
    h0 = torch.randn((N, H, K, V), dtype=rng_dtype)

    cu_seqlens = torch.tensor(np.cumsum([0] + seqlens_list), dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    # print("Running Triton Reference...")
    # h_ref, v_new_ref, final_state_ref = triton_fwd(
    #     k=k.float(), w=w.float(), u=u.float(), g=g.float(), gk=gk.float(),
    #     initial_state=h0.float(), output_final_state=True,
    #     chunk_size=chunk_size, save_new_value=True,
    #     cu_seqlens=cu_seqlens, use_exp2=False
    # )

    # 2. Pallas Implementation
    print("Running Pallas Implementation...")
    k_jax = jnp.array(k.to(torch.float32), dtype=pallas_dtype)
    w_jax = jnp.array(w.to(torch.float32), dtype=pallas_dtype)
    u_jax = jnp.array(u.to(torch.float32), dtype=pallas_dtype)
    g_jax = jnp.array(g.to(torch.float32), dtype=pallas_dtype)
    gk_jax = jnp.array(gk.to(torch.float32), dtype=pallas_dtype)
    h0_jax = jnp.array(h0.to(torch.float32), dtype=pallas_dtype)
    seqlens_jax = jnp.array(cu_seqlens.numpy(), dtype=jnp.int32)
    chunk_indices_jax = jnp.array(chunk_indices.numpy(), dtype=jnp.int32)

    h_history_jax, v_new_jax, final_state_jax = chunk_gated_delta_rule_fwd_h_varlen(
        k=k_jax, w=w_jax, u=u_jax, g=g_jax, gk=gk_jax,
        initial_state=h0_jax, output_final_state=True,
        chunk_size=chunk_size, save_new_value=True,
        seqlens=seqlens_jax, chunk_indices=chunk_indices_jax,
        use_exp2=False
    )
    jax.block_until_ready(final_state_jax)

    # 3. Compare
    # compare_tensor("Final State", final_state_ref, final_state_jax)
    # compare_tensor("Residual (v_new)", v_new_ref.squeeze(0), v_new_jax)
    # compare_tensor("Hidden History", h_ref.squeeze(0), h_history_jax)

if __name__ == "__main__":
  test_varlen()

