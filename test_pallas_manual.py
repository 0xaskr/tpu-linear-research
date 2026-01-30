import os
os.environ["TRITON_CPU_BACKEND"] = "1"
os.environ["TRITON_INTERPRET"] = "1"

from jax import config
config.update("jax_enable_x64", True)

from jax.experimental import pallas as pl
import jax.numpy as jnp
import jax
import functools
import numpy as np
import torch
import sys

sys.path.append(os.path.join(os.getcwd(), 'fla'))
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk as ref_chunk_gla_fwd_o_gk

def cdiv(a, b):
    return (a + b - 1) // b

def chunk_gla_fwd_o_gk_kernel(q_ref, v_ref, g_ref, h_ref, a_ref,
                              o_ref,
                              scale, T, H, K, V, BT, BK, BV, USE_EXP2):
  q = q_ref[:, :]
  g = g_ref[:, :]
  h = h_ref[0, 0, 0, :, :]
  if USE_EXP2:
    g_exp = jnp.exp2(g)
  else:
    g_exp = jnp.exp(g)
  qg = q * g_exp
  o_inter = jnp.dot(qg, h)
  o_inter = o_inter * scale
  a = a_ref[:, :]
  mask = jnp.tril(jnp.ones((BT, BT), dtype=bool))
  a = jnp.where(mask, a, 0).astype(a.dtype)
  v = v_ref[:, :]
  o_intra = jnp.dot(a, v)
  o = o_inter + o_intra
  o_ref[:, :] = o.astype(o_ref.dtype)

def chunk_gla_fwd_o_gk(q, v, g, a, h, scale, cu_seqlens=None, chunk_size=64, chunk_indices=None, use_exp2=False):
  B, T, H, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  NT = (T + BT - 1) // BT
  BK, BV = 64, 64
  assert q.shape == (B, T, H, K)
  assert v.shape == (B, T, H, V)
  assert g.shape == (B, T, H, K)
  assert a.shape == (B, T, H, BT)
  assert h.shape == (B, NT, H, K, V)
  q_blockspec = pl.BlockSpec([BT, K], index_map = lambda v, bt, bh: (bh, bt * BT, 0))
  v_blockspec = pl.BlockSpec([BT, BV], index_map = lambda v, bt, bh: (bh, bt * BT, v * BV))
  g_blockspec = pl.BlockSpec([BT, K], index_map = lambda v, bt, bh: (bh, bt * BT, 0))
  h_blockspec = pl.BlockSpec([1, 1, 1, K, BV], index_map = lambda v, bt, bh: (bh // H, bt, bh % H, 0, v * BV))
  a_block_spec = pl.BlockSpec([BT, BT], index_map = lambda v, bt, bh: (bh, bt * BT, 0))
  o_spec = jax.ShapeDtypeStruct([B * H, T, V], q.dtype)
  o_block_spec = pl.BlockSpec([BT, BV], index_map = lambda v, bt, bh: (bh, bt * BT, v * BV))
  grid = (cdiv(V, BV), NT, B * H)
  
  q = q.transpose(0, 2, 1, 3).reshape(B * H, T, K).squeeze(0)
  v = v.transpose(0, 2, 1, 3).reshape(B * H, T, V).squeeze(0)
  g = g.transpose(0, 2, 1, 3).reshape(B * H, T, K).squeeze(0)
  a = a.transpose(0, 2, 1, 3).reshape(B * H, T, BT).squeeze(0)
  
  # For h, we might need to be careful if we squeeze. 
  # Let's keep h's BlockSpec as 5D if we don't reshape it.
    functools.partial(
        chunk_gla_fwd_o_gk_kernel,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_EXP2=use_exp2,
    ),
    grid=grid,
    out_shape=o_spec,
    in_specs=[q_blockspec, v_blockspec, g_blockspec, h_blockspec, a_block_spec],
    out_specs=o_block_spec,
    interpret=True
  )(q, v, g, h, a)
  
  return o.reshape(B, H, T, V).transpose(0, 2, 1, 3)

if __name__ == "__main__":
  print("\n--- Testing chunk_gla_fwd_o_gk vs PyTorch ---")
  B, T, H, K, V = 1, 128, 1, 64, 64
  BT = 32
  NT = (T + BT - 1) // BT
  scale = K ** -0.5
  key = jax.random.PRNGKey(0)
  k1, k2, k3, k4, k5 = jax.random.split(key, 5)
  q_jax = jax.random.normal(k1, (B, T, H, K), dtype=jnp.float64)
  v_jax = jax.random.normal(k2, (B, T, H, V), dtype=jnp.float64)
  g_jax = -jax.random.exponential(k3, (B, T, H, K), dtype=jnp.float64) * 0.1
  A_jax = jax.random.normal(k4, (B, T, H, BT), dtype=jnp.float64)
  h_jax = jax.random.normal(k5, (B, NT, H, K, V), dtype=jnp.float64)
  try:
    o_jax = chunk_gla_fwd_o_gk(q_jax, v_jax, g_jax, A_jax, h_jax, scale, chunk_size=BT, use_exp2=False)
    o_jax.block_until_ready()
    print(f"JAX output NaNs: {np.isnan(o_jax).any()}")
    print(f"JAX output sample: {o_jax[0, 0, 0, :5]}")
  except Exception as e:
    print(f"JAX execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
  device = torch.device("cpu")
  print(f"Using device for PyTorch: {device}")
  q_pt = torch.tensor(np.array(q_jax), dtype=torch.float64).to(device)
  v_pt = torch.tensor(np.array(v_jax), dtype=torch.float64).to(device)
  g_pt = torch.tensor(np.array(g_jax), dtype=torch.float64).to(device)
  A_pt = torch.tensor(np.array(A_jax), dtype=torch.float64).to(device)
  h_pt = torch.tensor(np.array(h_jax), dtype=torch.float64).to(device)
  try:
    o_pt = ref_chunk_gla_fwd_o_gk(q_pt, v_pt, g_pt, A_pt, h_pt, scale, chunk_size=BT, use_exp2=False)
    print(f"PyTorch output NaNs: {torch.isnan(o_pt).any()}")
    print(f"PyTorch output sample: {o_pt[0, 0, 0, :5]}")
    o_pt_np = o_pt.cpu().detach().numpy()
    o_jax_np = np.array(o_jax)
    diff = np.abs(o_pt_np - o_jax_np)
    print(f"Max difference: {diff.max()}")
    print(f"Mean difference: {diff.mean()}")
    if diff.max() < 1e-3:
      print("Test PASSED!")
    else:
      print("Test FAILED!")
  except Exception as e:
    print(f"PyTorch execution failed: {e}")
    import traceback
    traceback.print_exc()