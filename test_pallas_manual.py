from jax.experimental import pallas as pl
import jax.numpy as jnp
import jax
import functools
import numpy as np

def cdiv(a, b):
    return (a + b - 1) // b

def chunk_gated_delta_rule_fwd(
    k_ref,
    v_ref,
    w_ref,
    g_ref,
    gk_ref,
    h0_ref,

    # output
    h_ref,
    v_new_ref,
    ht_ref,

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
  idx_v, idx_nh = pl.program_id(0), pl.program_id(1)
  idx_n, idx_h = idx_nh // H, idx_nh % H
  # h_ref[0, 0, 0, 0, 0] = jnp.ones([1], dtype=jnp.bfloat16)[0]
  # h_ref[0, 0, 0, 0, 0] = jnp.array(1.0, dtype=jnp.bfloat16)
  # h_ref[...] = jnp.arange(jnp.prod(jnp.array(h_ref.shape)), dtype=jnp.bfloat16).reshape(h_ref.shape)
  # h_ref[...] = jnp.arange(np.prod(h_ref.shape), dtype=jnp.bfloat16)
  # h_ref[...] = jnp.ones(h_ref.shape, dtype=h_ref.dtype)
  # v_new_ref[...] = jnp.ones(v_new_ref.shape, dtype=v_new_ref.dtype)
  # ht_ref[...] = jnp.ones(ht_ref.shape, dtype=ht_ref.dtype)

  bos = idx_n * T
  boh = idx_n * NT

  b_h1 = jnp.zeros([64, BV], dtype=jnp.float32)

  if K > 64:
      b_h2 = jnp.zeros([64, BV], dtype=jnp.float32)
  if K > 128:
      b_h3 = jnp.zeros([64, BV], dtype=jnp.float32)
  if K > 192:
      b_h4 = jnp.zeros([64, BV], dtype=jnp.float32)

  # h_ref += (boh * H + idx_h) * K * V
  # v_ref += (bos * H + idx_h) * V
  # k_ref += (bos * H + idx_h) * K
  # w_ref += (bos * H + idx_h) * K
  # h_ref = h_ref.at[idx_n, :, idx_h:, :, :]

  h_block_offset = jnp.array([idx_n, 0, idx_h, 0, 0], dtype = jnp.int32)
  h_block_shape = [1, NT, 1, K, V]

  v_block_offset = jnp.array([idx_n, 0, idx_h, 0], dtype = jnp.int32)
  v_block_shape = [1, T, 1, V]

  k_block_offset = jnp.array([idx_n, 0, idx_h, 0], dtype = jnp.int32)
  k_block_shape = [1, T, 1, K]

  w_block_offset = jnp.array([idx_n, 0, idx_h, 0], dtype = jnp.int32)
  w_block_shape = [1, T, 1, K]

  if SAVE_NEW_VALUE:
    v_new_block_offset = jnp.array([idx_n, 0, idx_h, 0], dtype = jnp.int32)
    v_new_block_shape = [1, T, 1, V]

  if USE_INITIAL_STATE:
    h0_block_offset = jnp.array([idx_n, idx_h, 0, 0], dtype = jnp.int32)
    h0_block_shape = [1, 1, K, V]

  if STORE_FINAL_STATE:
    ht_block_offset = jnp.array([idx_n, idx_h, 0, 0], dtype = jnp.int32)
    ht_block_shape = [1, 1, K, V]

  # TODO(0xaskr): support auto padding or padded tensor
  if USE_INITIAL_STATE:
    tmp_shape = [1, 1, 64, BV]
    b_h1_shape = [64, BV]
    tmp_offset = jnp.array([idx_n, idx_h, 0, idx_v * BV], dtype = jnp.int32)
    # b_h1 += jax.lax.dynamic_slice(h0_ref[...], tmp_offset, tmp_shape).astype(jnp.float32).reshape(b_h1_shape)
    b_h1 = h0_ref[idx_n, idx_h, :, idx_v * BV:idx_v * BV + BV]
    # if K > 64:
    #   tmp_offset = jnp.array([idx_n, idx_h, 64, idx_v * BV], dtype = jnp.int32)
    #   b_h2 += jax.lax.dynamic_slice(h0_ref[...], tmp_offset, tmp_shape).astype(jnp.float32).reshape(b_h1_shape)
    # if K > 128:
    #   tmp_offset = jnp.array([idx_n, idx_h, 128, idx_v * BV], dtype = jnp.int32)
    #   b_h3 += jax.lax.dynamic_slice(h0_ref[...], tmp_offset, tmp_shape).astype(jnp.float32).reshape(b_h1_shape)
    # if K > 128:
    #   tmp_offset = jnp.array([idx_n, idx_h, 192, idx_v * BV], dtype = jnp.int32)
    #   b_h4 += jax.lax.dynamic_slice(h0_ref[...], tmp_offset, tmp_shape).astype(jnp.float32).reshape(b_h1_shape)

  for i_t in range(NT):
    # h_ref[idx_n, i_t, idx_h, 0:64, idx_v * BV:idx_v * BV + 64] = b_h1.astype(h_ref.dtype).reshape(1, 1, 1, 64, BV)
    # h_ref: [B, NT, H, K, V]
    # b_h1: [64, BV]
    # We update a slice at [idx_n, i_t, idx_h, 0, idx_v * BV]

    start_indices = jnp.array([idx_n, i_t, idx_h, 0, idx_v * BV], dtype=jnp.int32)
    update_slice = b_h1.astype(h_ref.dtype).reshape(1, 1, 1, 64, BV)
    h_ref[...] = jax.lax.dynamic_update_slice(h_ref[...], update_slice, start_indices)


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

  B, H, T, K = k.shape
  V = u.shape[-1]
  BT = chunk_size
  NT = cdiv(T, BT)
  BV = 64

  h = jnp.zeros((B, NT, H, K, V), k.dtype)
  v_new = jnp.zeros_like(u) if save_new_value else None
  final_state = jnp.zeros((B, H, K, V), dtype=jnp.float32) if output_final_state else None

  h_spec = jax.ShapeDtypeStruct(h.shape, h.dtype)
  v_new_spec = jax.ShapeDtypeStruct(u.shape, u.dtype)
  final_state_spec = jax.ShapeDtypeStruct([B, H, K, V], jnp.float32)

  k_blockspec = pl.BlockSpec(k.shape, index_map = lambda x, y: (0, 0, 0, 0))
  u_blockspec = pl.BlockSpec(u.shape, index_map = lambda x, y: (0, 0, 0, 0))
  w_blockspec = pl.BlockSpec(w.shape, index_map = lambda x, y: (0, 0, 0, 0))
  g_blockspec = pl.BlockSpec(g.shape, index_map = lambda x, y: (0, 0, 0))
  gk_blockspec = pl.BlockSpec(gk.shape, index_map = lambda x, y: (0, 0, 0, 0))
  init_blockspec = pl.BlockSpec([B, H, K, V], index_map = lambda x, y: (0, 0, 0, 0))

  h_blockspec = pl.BlockSpec(h.shape, lambda x, y : (0, 0, 0, 0, 0))
  v_new_blockspec = pl.BlockSpec(u.shape, lambda x, y : (0, 0, 0, 0))
  final_out_blockspec = pl.BlockSpec([B, H, K, V], lambda x, y : (0, 0, 0, 0))

  # 如果遇到输入输出 可选的情况, 必须要给进去吗?
  grid = ((V + BV - 1) // BV, B * H)
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
  )(k, u, w, g, gk, initial_state)

  return h, (v_out if save_new_value else v_new), (final_out if output_final_state else final_state)


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
