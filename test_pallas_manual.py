from jax.experimental import pallas as pl
import jax.numpy as jnp
import jax
import functools
import numpy as np

def cdiv(a, b):
    return (a + b - 1) // b

# TODO(lain.shen) need h0, k pad to align up 64, pad 0
# TODO(lain.shen) h pad to 8
def chunk_gated_delta_rule_fwd(
    k_ref,  # [1, 1, T, K]
    v_ref,  # [1, 1, T, BV]
    w_ref,  # [1, 1, T, K]
    g_ref,  # [B, T, H]
    gk_ref, # [B, T, H, K]
    h0_ref, # [1, 1, K, BV]

    # output
    h_ref,  # [1, NT, 1, K, BV]
    v_new_ref,  # [1, 1, T, BV]
    ht_ref,     # [1, 1, K, BV]

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
):
  idx_v, idx_n, idx_h = pl.program_id(0), pl.program_id(1), pl.program_id(2)

  b_h1 = jnp.zeros([64, BV], dtype=jnp.float32)
  if K > 64:
    b_h2 = jnp.zeros([64, BV], dtype=jnp.float32)
  if K > 128:
    b_h3 = jnp.zeros([64, BV], dtype=jnp.float32)
  if K > 192:
    b_h4 = jnp.zeros([64, BV], dtype=jnp.float32)


  # TODO(0xaskr): support auto padding or padded tensor
  if USE_INITIAL_STATE:
    b_h1 += h0_ref[0, 0, 0:64, 0:BV].astype(jnp.float32)
    if K > 64:
      b_h2 += h0_ref[0, 0, 64:128, 0:BV].astype(jnp.float32) # type: ignore
    if K > 128:
      b_h3 += h0_ref[0, 0, 128:192, 0:BV].astype(jnp.float32) # type: ignore
    if K > 192:
      b_h4 += h0_ref[0, 0, 192:256, 0:BV].astype(jnp.float32) # type: ignore

  for i_t in range(NT):
    h_ref[0, i_t, 0, 0:64, 0:BV] = b_h1.astype(h_ref.dtype)
    if K > 64:
      h_ref[0, i_t, 0, 64:128, 0:BV] = b_h2.astype(h_ref.dtype) # type: ignore
    if K > 128:
      h_ref[0, i_t, 0, 128:192, 0:BV] = b_h3.astype(h_ref.dtype)  # type: ignore
    if K > 192:
      h_ref[0, i_t, 0, 192:256, 0:BV] = b_h4.astype(h_ref.dtype)  # type: ignore

    b_w = w_ref[0, 0, i_t * BT: i_t * BT + BT, 0:64]
    b_v = jnp.dot(b_w, b_h1.astype(b_w.dtype), preferred_element_type=jnp.float32)
    # b_v = jnp.dot(b_w.astype(jnp.float32), b_h1, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    if (K > 64):
      b_w = w_ref[0, 0, i_t * BT: i_t * BT + BT, 64:128]
      b_v += jnp.dot(b_w, b_h2.astype(b_w.dtype), preferred_element_type=jnp.float32)
    if (K > 128):
      b_w = w_ref[0, 0, i_t * BT: i_t * BT + BT, 128:192]
      b_v += jnp.dot(b_w, b_h3.astype(b_w.dtype), preferred_element_type=jnp.float32)
    if (K > 192):
      b_w = w_ref[0, 0, i_t * BT: i_t * BT + BT, 192:256]
      b_v += jnp.dot(b_w, b_h4.astype(b_w.dtype), preferred_element_type=jnp.float32)
    b_v = v_ref[0, 0, i_t * BT: i_t * BT + BT, 0:BV].astype(b_v.dtype) - b_v

    if (SAVE_NEW_VALUE):
      print("v_new_dtype = ", v_new_ref.dtype, b_v.dtype)
      v_new_ref[0, 0, i_t * BT: i_t * BT + BT, 0:BV] = b_v.astype(v_new_ref.dtype)

    # 如果证明, T被pad到BT的整数倍, 这里
    # assert T % BT == 0
    last_idx = min((i_t + 1) * BT, T) - 1

    if (USE_G):
      m_t = (i_t * BT + jnp.arange(0, BT)) < T
      b_g_last = g_ref[0, idx_h, last_idx].astype(jnp.float32)
      b_g = g_ref[0, idx_h, i_t * BT: i_t * BT + BT].astype(jnp.float32)
      if USE_EXP2:
        b_v = b_v * jnp.where(m_t, jnp.exp2(b_g_last - b_g), 0)[:,None]
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
      b_gk_last1 = jnp.where(o_k1 < K, gk_ref[0, 0, last_idx, 0:64], 0).astype(jnp.float32)
      if USE_EXP2:
        b_h1 *= jnp.exp2(b_gk_last1)[:, None]
      else:
        b_h1 *= jnp.exp(b_gk_last1)[:, None]
      if K > 64:
        o_k2 = 64 + o_k1
        b_gk_last2 = jnp.where(o_k2 < K, gk_ref[0, 0, last_idx, 64:128], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h2 *= jnp.exp2(b_gk_last2)[:, None]
        else:
          b_h2 *= jnp.exp(b_gk_last2)[:, None]
      if K > 128:
        o_k3 = 128 + o_k1
        b_gk_last3 = jnp.where(o_k3 < K, gk_ref[0, 0, last_idx, 128:192], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h3 *= jnp.exp2(b_gk_last3)[:, None]
        else:
          b_h3 *= jnp.exp(b_gk_last3)[:, None]
      if K > 192:
        o_k4 = 192 + o_k1
        b_gk_last4 = jnp.where(o_k4 < K, gk_ref[0, 0, last_idx, 192:256], 0).astype(jnp.float32)
        if USE_EXP2:
          b_h4 *= jnp.exp2(b_gk_last4)[:, None]
        else:
          b_h4 *= jnp.exp(b_gk_last4)[:, None]

    # b_v = b_v.astype(k_ref.dtype)

    b_k = k_ref[0, 0, i_t * BT:i_t * BT + BT, 0:64].reshape(BT, 64).transpose(1, 0)
    b_h1 += jnp.dot(b_k.astype(jnp.float32), b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 64:
      b_k = k_ref[0, 0, i_t * BT:i_t * BT + BT, 64:128].reshape(BT, 64).transpose(1, 0)
      b_h2 += jnp.dot(b_k.astype(jnp.float32), b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 128:
      b_k = k_ref[0, 0, i_t * BT:i_t * BT + BT, 128:192].reshape(BT, 64).transpose(1, 0)
      b_h3 += jnp.dot(b_k.astype(jnp.float32), b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    if K > 192:
      b_k = k_ref[0, 0, i_t * BT:i_t * BT + BT, 192:256].reshape(BT, 64).transpose(1, 0)
      b_h4 += jnp.dot(b_k.astype(jnp.float32), b_v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

  if STORE_FINAL_STATE:
    ht_ref[0, 0, 0:64, 0:BV] = b_h1.astype(ht_ref.dtype)
    if K > 64:
      ht_ref[0, 0, 64:128, 0:BV] = b_h2.astype(ht_ref.dtype)
    if K > 128:
      ht_ref[0, 0, 128:192, 0:BV] = b_h3.astype(ht_ref.dtype)
    if K > 192:
      ht_ref[0, 0, 192:256, 0:BV] = b_h4.astype(ht_ref.dtype)

def chunk_gated_delta_rule_fwd_h(
    k: jax.Array,
    w: jax.Array,
    u: jax.Array,
    g: jax.Array | None = None,
    gk: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    use_exp2: bool = False,
):

  B, T, H, K = k.shape
  V = u.shape[-1]
  BT = chunk_size
  NT = cdiv(T, BT)
  BV = 64

  assert k.shape == (B, T, H, K)
  assert w.shape == (B, T, H, K)
  assert u.shape == (B, T, H, V)
  if g is not None:
    assert g.shape == (B, T, H)
  if gk is not None:
    assert gk.shape == (B, T, H, K)
  if (initial_state is not None):
     assert initial_state.shape == (B, H, K, V)

  h = jnp.zeros((B, NT, H, K, V), k.dtype)
  v_new = jnp.zeros_like(u) if save_new_value else None
  final_state = jnp.zeros((B, H, K, V), dtype=jnp.float32) if output_final_state else None

  h_spec = jax.ShapeDtypeStruct(h.shape, h.dtype)
  v_new_spec = jax.ShapeDtypeStruct([B, H, T, V], u.dtype)
  final_state_spec = jax.ShapeDtypeStruct([B, H, K, V], jnp.float32)

  k = jnp.transpose(k, [0, 2, 1, 3])    # [B, T, H, K] -> [B, H, T, K]
  u = jnp.transpose(u, [0, 2, 1, 3])    # [B, T, H, V] -> [B, H, T, V]
  w = jnp.transpose(w, [0, 2, 1, 3])    # [B, T, H, K] -> [B, H, T, K]
  if g is not None:
    g = jnp.transpose(g, [0, 2, 1])       # [B, T, H]    -> [B, H, T]
    g_fp32 = g.astype(jnp.float32)
  else:
    g_fp32 = None

  if gk is not None:
    gk = jnp.transpose(gk, [0, 2 ,1 ,3])  # [B, T, H, K] -> [B, H, T, K]
    gk_fp32 = gk.astype(jnp.float32)
  else:
    gk_fp32 = None

  # v_new = jnp.transpose(v_new, [0, 2, 1, 3])  # [B, T, H, V] -> [B, H, T, V]

  k_blockspec = pl.BlockSpec([1, 1, T, K], index_map = lambda v, b, h: (b, h, 0, 0))        # need trans
  u_blockspec = pl.BlockSpec([1, 1, T, BV], index_map = lambda v, b, h: (b, h, 0, v * BV))  # need trans
  w_blockspec = pl.BlockSpec([1, 1, T, K], index_map = lambda v, b, h: (b, h, 0, 0))        # need trans
  g_blockspec = pl.BlockSpec([1, H, T], index_map = lambda v, b, h: (b, 0, 0))
  gk_blockspec = pl.BlockSpec([1, 1, T, K], index_map = lambda v, b, h: (b, h, 0, 0))

  h_blockspec = pl.BlockSpec([1, NT, 1, K, BV], lambda v, b, h : (b, 0, h, 0, v * BV))
  v_new_blockspec = pl.BlockSpec([1, 1, T, BV], lambda v, b, h : (b, h, 0, v * BV))         # need trans
  init_blockspec = pl.BlockSpec([1, 1, K, BV], index_map = lambda v, b, h: (b, h, 0, v * BV))
  final_out_blockspec = pl.BlockSpec([1, 1, K, BV], lambda v, b, h : (b, h, 0, v * BV))

  # 如果遇到输入输出 可选的情况, 必须要给进去吗?
  # TODO(lain.shen) 使用B, H, V
  grid = ((V + BV - 1) // BV, B, H)
  h, v_out, final_out = pl.pallas_call(
    functools.partial(
        chunk_gated_delta_rule_fwd,
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
    ),
    grid=grid,
    out_shape=[h_spec, v_new_spec, final_state_spec],
    in_specs=[k_blockspec, u_blockspec, w_blockspec, g_blockspec if (g is not None) else None, gk_blockspec if (gk is not None) else None, init_blockspec],
    out_specs=[h_blockspec, v_new_blockspec, final_out_blockspec],
    # interpret=True
  )(k, u, w, g_fp32, gk_fp32, initial_state)

  v_out = jnp.transpose(v_out, [0, 2, 1, 3]) # [B, H, T, V] -> [B, T, H, V]

  return h, (v_out if save_new_value else None), (final_out if output_final_state else None)


def chunk_gla_fwd_o_gk_kernel(q_ref, v_ref, g_ref, h_ref, a_ref,
                              o_ref,
                              scale, T, H, K, V, BT, BK, BV, USE_EXP2):
  idx_v, idx_t, idx_b, idx_h = pl.program_id(0), pl.program_id(1), pl.program_id(2), pl.program_id(3)
  NT = cdiv(T, BT)
  idx_tg = idx_b * NT + idx_t

  m_s = jnp.arange(0, BT)[:, None] >= jnp.arange(0, BT)[None, :]
  b_o = jnp.zeros([BT, BV], dtype=jnp.float32)

  for idx_k in range(cdiv(K, BK)):
    q = q_ref[...].reshape(BT, K)[:, idx_k * BK: (idx_k + 1) * BK]
    g = g_ref[...].reshape(BT, K)[:, idx_k * BK: (idx_k + 1) * BK]
    h = h_ref[...].reshape(K, BV)[idx_k * BK: (idx_k + 1) * BK, :]
    # 这里是决定精度的重点
    if USE_EXP2:
      qg = (q.astype(jnp.float32) * jnp.exp2(g.astype(jnp.float32))).astype(q.dtype)
    else:
      qg = (q.astype(jnp.float32) * jnp.exp(g.astype(jnp.float32))).astype(q.dtype)

    if idx_k >= 0:
      b_o += jnp.dot(qg, h.astype(qg.dtype), preferred_element_type=jnp.float32)
      # b_o += jnp.dot(qg.astype(jnp.float32), h.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

  b_o *= scale
  v = v_ref[...].reshape(BT, BV)
  a = a_ref[...].reshape(BT, BT)
  a = jnp.where(m_s, a, 0).astype(v.dtype)
  b_o += jnp.dot(a, v, preferred_element_type=jnp.float32)
  # b_o += jnp.dot(a.astype(jnp.float32), v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
  o_ref[...] = b_o.reshape(1, 1, BT, BV).astype(o_ref.dtype)


def chunk_gla_fwd_o_gk(q : jax.Array,
                       v : jax.Array,
                       g : jax.Array,
                       a : jax.Array,
                       h : jax.Array,
                       scale: float,
                       cu_seqlens: jax.Array | None = None,
                       chunk_size: int = 64,
                       chunk_indices: jax.Array | None = None,
                       use_exp2: bool = False) -> jax.Array:
  """
  Reference implementation for GLA chunk forward output.
  o = (q * exp(g)) @ h * scale + A @ v
  """
  B, T, H, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = cdiv(T ,BT) if cu_seqlens is None else len(chunk_indices)
  BK, BV = 32, 64

  assert q.shape == (B, T, H, K)
  assert v.shape == (B, T, H, V)
  assert g.shape == (B, T, H, K)
  assert a.shape == (B, T, H, BT)
  assert h.shape == (B, NT, H, K, V)

  if cu_seqlens is not None:
    pass
  if chunk_indices is not None:
    pass

  # Transpose inputs to satisfy Pallas TPU alignment (dim -2 % 8 == 0)
  # (B, T, H, K) -> (B, H, T, K)
  q = jnp.transpose(q, (0, 2, 1, 3))
  v = jnp.transpose(v, (0, 2, 1, 3))
  g = jnp.transpose(g, (0, 2, 1, 3))
  a = jnp.transpose(a, (0, 2, 1, 3))

  # New BlockSpecs for (B, H, T, ...) layout
  q_blockspec = pl.BlockSpec([1, 1, BT, K], index_map = lambda v, bt, b, h: (b, h, bt * BT, 0))
  v_blockspec = pl.BlockSpec([1, 1, BT, BV], index_map = lambda v, bt, b, h: (b, h, bt * BT, v * BV))
  g_blockspec = pl.BlockSpec([1, 1, BT, K], index_map = lambda v, bt, b, h: (b, h, bt * BT, 0))
  # h stays (B, NT, H, K, V)
  h_blockspec = pl.BlockSpec([1, 1, 1, K, BV], index_map = lambda v, bt, b, h: (b, bt, h, 0, v * BV))
  a_block_spec = pl.BlockSpec([1, 1, BT, BT], index_map = lambda v, bt, b, h: (b, h, bt * BT, 0))

  # TODO(baihua): use bh
  # Output (B, H, T, V)
  o_spec = jax.ShapeDtypeStruct([B, H, T, V], v.dtype)
  o_block_spec = pl.BlockSpec([1, 1, BT, BV], index_map = lambda v, bt, b, h: (b, h, bt * BT, v * BV))
  grid = (cdiv(V, BV), NT, B, H)
  o = pl.pallas_call(
    functools.partial(
        chunk_gla_fwd_o_gk_kernel,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK = BK,
        BV=BV,
        USE_EXP2=use_exp2,
    ),
    grid=grid,
    out_shape=[o_spec],
    in_specs=[q_blockspec, v_blockspec, g_blockspec, h_blockspec, a_block_spec],
    out_specs=[o_block_spec],
    # interpret=True
  )(q, v, g, h, a)

  if isinstance(o, (tuple, list)):
    o = o[0]
  o = jnp.transpose(o, (0, 2, 1, 3))
  return o


if __name__ == "__main__":
  B, T, H, K, V = 2, 128, 4, 64, 64
  chunk_size = 64
  use_exp2 = True
  device = "cpu"

  k_shape = [B, T, H, K]
  w_shape = [B, T, H, K]
  u_shape = [B, T, H, V]
  g_shape = [B, T, H]
  gk_shape = [B, T, H, K]
  h0_shape = [B, H, K, V]

  k = jnp.arange(jnp.prod(jnp.array(k_shape))).reshape(k_shape).astype(jnp.bfloat16)
  w = jnp.arange(jnp.prod(jnp.array(w_shape))).reshape(w_shape).astype(jnp.bfloat16)
  u = jnp.arange(jnp.prod(jnp.array(u_shape))).reshape(u_shape).astype(jnp.bfloat16)
  g = jnp.arange(jnp.prod(jnp.array(g_shape))).reshape(g_shape).astype(jnp.bfloat16)
  gk = (
      jnp.arange(jnp.prod(jnp.array(gk_shape))).reshape(gk_shape).astype(jnp.bfloat16)
  )
  h0 = (
      jnp.arange(jnp.prod(jnp.array(h0_shape))).reshape(h0_shape).astype(jnp.bfloat16)
  )

  h, v_new, final_out = chunk_gated_delta_rule_fwd_h(
      k, w, u, g, gk, h0, output_final_state=True, chunk_size=64, save_new_value=True
  )
  print("h = ", h.reshape(-1)[:10])
  print("v_new = ", v_new.reshape(-1)[:10] if v_new is not None else None)
  print("final_out = ", final_out.reshape(-1)[:10] if final_out is not None else None)