from jax.experimental import pallas as pl
import jax.numpy as jnp
import jax
import functools
import numpy as np

def chunk_gated_delta_rule_fwd(
  k_ref,
  v_ref,
  w_ref,
  v_new_ref,
  g_ref,
  gk_ref,
  h_ref,
  h0_ref,
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
  USE_EXP2
):
  idx_v, idx_nh = pl.program_id(0), pl.program_id(1)
  idx_n, idx_h = idx_nh // H, idx_nh % H

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
  h_ref = h_ref.at[idx_n,:,idx_h:,:,:]


  if (SAVE_NEW_VALUE):
    v_new_ref += (bos * H + idx_h) * V
  
  if USE_INITIAL_STATE:
    h0 = h0_ref + idx_nh * K * V
  
  if STORE_FINAL_STATE:
    ht = ht_ref + idx_nh * K * V
  

if __name__ == "__main__":
  B, T, H, K, V = 2, 128, 4, 32, 64
  chunk_size = 64
  use_exp2 = True
  device = 'cpu'

  k_shape = [B, T, H, K]
  w_shape = [B, T, H, K]
  u_shape = [B, T, H, V]
  g_shape = [B, T, H]
  gk_shape = [B, T, H, K]
  h0_shape = [B, H, K, V]


  k_torch = jnp.arange(jnp.prod(jnp.array(k_shape))).reshape(k_shape).astype(jnp.bfloat16)
  w_torch = jnp.arange(jnp.prod(jnp.array(w_shape))).reshape(w_shape).astype(jnp.bfloat16)
  u_torch = jnp.arange(jnp.prod(jnp.array(u_shape))).reshape(u_shape).astype(jnp.bfloat16)
  g_torch = jnp.arange(jnp.prod(jnp.array(g_shape))).reshape(g_shape).astype(jnp.bfloat16)
  gk_torch = jnp.arange(jnp.prod(jnp.array(gk_shape))).reshape(gk_shape).astype(jnp.bfloat16)
  h0_torch = jnp.arange(jnp.prod(jnp.array(h0_shape))).reshape(h0_shape).astype(jnp.bfloat16)
  
  


