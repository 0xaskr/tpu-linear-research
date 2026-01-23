
import os
import sys
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from delta_attention_comparison.src.layers.pallas_kda import kda_intra_chunk_fwd

def compute_chunk_vars_ref(k_blk, g_blk, beta_blk, v_blk, chunk_size=128):
    """
    Reference implementation of KDA intra-chunk computation.
    
    Args:
        k_blk: (C, D)
        g_blk: (C, D) - cumulative sum of logs
        beta_blk: (C,)
        v_blk: (C, D)
    Returns:
        u: (C, D)
        w: (C, D)
    """
    prec = jax.lax.Precision.HIGHEST

    # g_diff: (C, C, D)
    # g_diff[i, j] = g[i] - g[j]
    g_diff = g_blk[:, None, :] - g_blk[None, :, :]
    
    idx = jnp.arange(chunk_size)
    mask = idx[:, None] > idx[None, :] 
    
    # Mask positive exponents (j > i) to avoid overflow
    safe_g_diff = jnp.where(jnp.expand_dims(mask, -1), g_diff, -float('inf'))
    
    # term: (C, C, D)
    term = (k_blk[:, None, :] * k_blk[None, :, :]) * jnp.exp(safe_g_diff)
    
    A_raw = jnp.sum(term, axis=-1)

    A = A_raw * jnp.expand_dims(beta_blk, -1)
    
    # T = (I + A)^{-1}
    eye = jnp.eye(chunk_size, dtype=A.dtype)
    L = eye + A
    T = jax.scipy.linalg.solve_triangular(L, eye, lower=True)
    
    # T_final = T * diag(beta)
    # T_final[i, :] = T[i, :] * beta[i]
    T_final = T * jnp.expand_dims(beta_blk, -1) 
    
    u = jnp.matmul(T_final, v_blk, precision=prec)
    w = jnp.matmul(T_final, k_blk * jnp.exp(g_blk), precision=prec)
    
    return u, w

# Vmap over Batch, Heads, and Chunks
compute_chunk_ref_vmap = jax.vmap(jax.vmap(jax.vmap(compute_chunk_vars_ref, in_axes=(0,0,0,0,None)), in_axes=(0,0,0,0,None)), in_axes=(0,0,0,0,None))

class TestPallasKDA(unittest.TestCase):
    def test_intra_chunk_fwd(self):
        print(f"JAX Backend: {jax.default_backend()}")
        print(f"JAX Devices: {jax.devices()}")
        
        # Config
        B, H, T, D = 1, 2, 256, 128
        chunk_size = 128
        dtype = jnp.float32
        
        # Seeds
        key = random.PRNGKey(0)
        k1, k2, k3, k4 = random.split(key, 4)
        
        # Init inputs
        k = random.normal(k1, (B, H, T, D), dtype=dtype)
        # g is log-sigmoid, so negative values. cumsum makes them decreasing.
        g_raw = jax.nn.log_sigmoid(random.normal(k2, (B, H, T, D), dtype=dtype))
        
        # In the original code, g passed to compute_chunk_vars is cumsum within the chunk?
        # No, kimi_delta_attention.py does:
        # g_c = to_chunk(g) ... g_cumsum = jnp.cumsum(g_c, axis=-2)
        # compute_chunk_vars(..., g_cumsum, ...)
        # So g passed to kernel is cumulative sum from the start of the chunk.
        
        # Reshape to chunks to compute cumsum correctly per chunk
        num_chunks = T // chunk_size
        g_reshaped = g_raw.reshape(B, H, num_chunks, chunk_size, D)
        g_cumsum = jnp.cumsum(g_reshaped, axis=-2)
        # Flatten back to (B, H, T, D) for the pallas kernel interface (which handles reshaping inside)
        g_in = g_cumsum.reshape(B, H, T, D)
        
        beta = jax.nn.sigmoid(random.normal(k3, (B, H, T), dtype=dtype))
        v = random.normal(k4, (B, H, T, D), dtype=dtype)
        
        # Run Reference
        # Reference expects chunks: (B, H, num_chunks, chunk_size, D)
        k_c = k.reshape(B, H, num_chunks, chunk_size, D)
        beta_c = beta.reshape(B, H, num_chunks, chunk_size)
        v_c = v.reshape(B, H, num_chunks, chunk_size, D)
        g_c_ref = g_cumsum # Already chunked and summed
        
        print("Running Reference...")
        u_ref_c, w_ref_c = compute_chunk_ref_vmap(k_c, g_c_ref, beta_c, v_c, chunk_size)
        u_ref = u_ref_c.reshape(B, H, T, D)
        w_ref = w_ref_c.reshape(B, H, T, D)
        
        # Run Pallas
        print("Running Pallas...")
        try:
            u_pallas, w_pallas = kda_intra_chunk_fwd(k, g_in, beta, v, chunk_size=chunk_size)
        except Exception as e:
            print(f"Pallas execution failed (expected if not on TPU): {e}")
            # Skip assertion if Pallas fails (e.g. on CPU)
            if jax.default_backend() == 'cpu':
                print("Skipping Pallas assertion on CPU.")
                return
            else:
                raise e

        # Check
        print("Comparing results...")
        diff_u = jnp.max(jnp.abs(u_ref - u_pallas))
        diff_w = jnp.max(jnp.abs(w_ref - w_pallas))
        
        print(f"Max Diff U: {diff_u}")
        print(f"Max Diff W: {diff_w}")
        
        # Tolerances
        atol = 2e-3 if dtype == jnp.float32 else 1e-2
        rtol = 2e-3 if dtype == jnp.float32 else 1e-2
        
        np.testing.assert_allclose(u_pallas, u_ref, atol=atol, rtol=rtol, err_msg="U mismatch")
        np.testing.assert_allclose(w_pallas, w_ref, atol=atol, rtol=rtol, err_msg="W mismatch")
        print("Test Passed!")

if __name__ == '__main__':
    unittest.main()
