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
    b_v = jnp.dot(b_w.astype(jnp.float32), b_h1)
    if (K > 64):
      b_w = w_ref[0, 0, i_t * BT: i_t * BT + BT, 64:128]
      b_v += jnp.dot(b_w.astype(jnp.float32), b_h2)
    if (K > 128):
      b_w = w_ref[0, 0, i_t * BT: i_t * BT + BT, 128:192]
      b_v += jnp.dot(b_w.astype(jnp.float32), b_h3)
    if (K > 128):
      b_w = w_ref[0, 0, i_t * BT: i_t * BT + BT, 192:256]
      b_v += jnp.dot(b_w.astype(jnp.float32), b_h4)
    b_v = b_v.astype(b_w.dtype)

    b_v = v_ref[0, 0, i_t * BT: i_t * BT + BT, 0:BV] - b_v

    if (SAVE_NEW_VALUE):
      v_new_ref[0, 0, i_t * BT: i_t * BT + BT, 0:BV] = b_v.astype(v_new_ref.dtype)

    # 如果证明, T被pad到BT的整数倍, 这里
    # assert T % BT == 0
    last_idx = min((i_t + 1) * BT, T) - 1

    if (USE_G):
      m_t = (i_t * BT + jnp.arange(0, BT)) < T
      b_g_last = g_ref[0, idx_h, last_idx]
      b_g = g_ref[0, idx_h, i_t * BT: i_t * BT + BT]
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
      b_gk_last1 = jnp.where(o_k1 < K, gk_ref[0, 0, last_idx, 0:64], 0)
      if USE_EXP2:
        b_h1 *= jnp.exp2(b_gk_last1)[:, None]
      else:
        b_h1 *= jnp.exp(b_gk_last1)[:, None]
      if K > 64:
        o_k2 = 64 + o_k1
        b_gk_last2 = jnp.where(o_k2 < K, gk_ref[0, 0, last_idx, 64:128], 0)
        if USE_EXP2:
          b_h2 *= jnp.exp2(b_gk_last2)[:, None]
        else:
          b_h2 *= jnp.exp(b_gk_last2)[:, None]
      if K > 128:
        o_k3 = 128 + o_k1
        b_gk_last3 = jnp.where(o_k3 < K, gk_ref[0, 0, last_idx, 128:192], 0)
        if USE_EXP2:
          b_h3 *= jnp.exp2(b_gk_last3)[:, None]
        else:
          b_h3 *= jnp.exp(b_gk_last3)[:, None]
      if K > 192:
        o_k4 = 192 + o_k1
        b_gk_last4 = jnp.where(o_k4 < K, gk_ref[0, 0, last_idx, 192:256], 0)
        if USE_EXP2:
          b_h4 *= jnp.exp2(b_gk_last4)[:, None]
        else:
          b_h4 *= jnp.exp(b_gk_last4)[:, None]

    b_v = b_v.astype(k_ref.dtype)

    b_k = k_ref[0, 0, i_t * BT:i_t * BT + BT, 0:64].reshape(BT, 64).transpose(1, 0)
    b_h1 += jnp.dot(b_k, b_v, preferred_element_type=jnp.float32)
    if K > 64:
      b_k = k_ref[0, 0, i_t * BT:i_t * BT + BT, 64:128].reshape(BT, 64).transpose(1, 0)
      b_h2 += jnp.dot(b_k, b_v)
    if K > 128:
      b_k = k_ref[0, 0, i_t * BT:i_t * BT + BT, 128:192].reshape(BT, 64).transpose(1, 0)
      b_h3 += jnp.dot(b_k, b_v)
    if K > 192:
      b_k = k_ref[0, 0, i_t * BT:i_t * BT + BT, 192:256].reshape(BT, 64).transpose(1, 0)
      b_h4 += jnp.dot(b_k, b_v)

  if STORE_FINAL_STATE:
    ht_ref[0, 0, 0:64, 0:BV] = b_h1.astype(ht_ref.dtype)
    if K > 64:
      ht_ref[0, 0, 64:128, 0:BV] = b_h2.astype(ht_ref.dtype)
    if K > 128:
      ht_ref[0, 0, 128:192, 0:BV] = b_h3.astype(ht_ref.dtype)
    if K > 192:
      ht_ref[0, 0, 192:256, 0:BV] = b_h4.astype(ht_ref.dtype)

def kernel_wrapper(
    k: jax.Array,

    w: jax.Array,
    u: jax.Array,
    g: jax.Array,
    gk: jax.Array,
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
  assert g.shape == (B, T, H)
  assert gk.shape == (B, T, H, K)
  if (initial_state is not None):
     assert initial_state.shape == (B, H, K, V)

  h = jnp.zeros((B, NT, H, K, V), k.dtype)
  v_new = jnp.zeros_like(u) if save_new_value else None
  final_state = jnp.zeros((B, H, K, V), dtype=jnp.float32) if output_final_state else None

  h_spec = jax.ShapeDtypeStruct(h.shape, h.dtype)
  v_new_spec = jax.ShapeDtypeStruct(u.shape, u.dtype)
  final_state_spec = jax.ShapeDtypeStruct([B, H, K, V], jnp.float32)

  k = jnp.transpose(k, [0, 2, 1, 3])    # [B, T, H, K] -> [B, H, T, K]
  u = jnp.transpose(u, [0, 2, 1, 3])    # [B, T, H, V] -> [B, H, T, V]
  w = jnp.transpose(w, [0, 2, 1, 3])    # [B, T, H, K] -> [B, H, T, K]
  g = jnp.transpose(g, [0, 2, 1])       # [B, T, H]    -> [B, H, T]
  gk = jnp.transpose(gk, [0, 2 ,1 ,3])  # [B, T, H, K] -> [B, H, T, K]
  g_fp32 = g.astype(jnp.float32)
  gk_fp32 = gk.astype(jnp.float32)
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
    in_specs=[k_blockspec, u_blockspec, w_blockspec, g_blockspec, gk_blockspec, init_blockspec],
    out_specs=[h_blockspec, v_new_blockspec, final_out_blockspec],
  )(k, u, w, g_fp32, gk_fp32, initial_state)

  v_out = jnp.transpose(v_out, [0, 2, 1, 3]) # [B, H, T, V] -> [B, T, H, V]

  return h, (v_out if save_new_value else None), (final_out if output_final_state else None)


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

  h, v_new, final_out = kernel_wrapper(
      k, w, u, g, gk, h0, output_final_state=True, chunk_size=64, save_new_value=True
  )
  print("h = ", h.reshape(-1)[:10])
  print("v_new = ", v_new.reshape(-1)[:10] if v_new is not None else None)
  print("final_out = ", final_out.reshape(-1)[:10] if final_out is not None else None)
