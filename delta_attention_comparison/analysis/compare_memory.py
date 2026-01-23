import argparse
import sys
import os
import jax
import jax.numpy as jnp
from flax import nnx

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.layers.kimi_delta_attention import KimiDeltaAttention as KimiNew

def analyze_peak_memory(model_class, name, batch_size, seq_len, hidden_size, num_heads, head_dim):
    print(f"Analyzing {name} (B={batch_size}, L={seq_len})...")
    
    try:
        from jax.experimental import roofline
    except ImportError:
        print("jax.experimental.roofline not available.")
        return 0.0

    rngs = nnx.Rngs(0)
    model = model_class(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=jnp.bfloat16,
        rngs=rngs
    )
    
    # We analyze the full forward pass call
    graph, params = nnx.split(model)
    
    def full_call(p, x):
        m = nnx.merge(graph, p)
        return m(x)[0]
    
    abstract_params = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)
    input_shape = jax.ShapeDtypeStruct((batch_size, seq_len, hidden_size), jnp.bfloat16)
    
    # Analyze Forward Peak
    _, res = roofline.roofline(full_call)(abstract_params, input_shape)
    
    peak_mb = res.peak_hbm_bytes / 1e6
    print(f"  [{name}] Peak Memory: {peak_mb:.2f} MB")
    
    # Also verify Backward Peak (where activation checkpointing/rematerialization usually matters)
    def loss_fn(p, x):
        return jnp.sum(full_call(p, x))
        
    grad_fn = jax.grad(loss_fn)
    try:
        _, res_bwd = roofline.roofline(grad_fn)(abstract_params, input_shape)
        peak_bwd_mb = res_bwd.peak_hbm_bytes / 1e6
        print(f"  [{name}] Backward Peak: {peak_bwd_mb:.2f} MB")
    except Exception as e:
        print(f"  [{name}] Backward Analysis failed: {e}")
        peak_bwd_mb = 0.0
        
    return peak_mb, peak_bwd_mb

def main():
    # Large sequence length to make the memory difference obvious
    B = 1
    L = 32768 # 32k context
    H = 2048
    NH = 16
    D = 128
    
    print("=== Memory Optimization Verification ===")
    
    peak_new_fwd, peak_new_bwd = analyze_peak_memory(KimiNew, "Optimized", B, L, H, NH, D)
    
    print("\n=== Summary ===")
    print(f"Forward Peak: {peak_new_fwd:.2f} MB")
    print(f"Backward Peak: {peak_new_bwd:.2f} MB")

if __name__ == "__main__":
    main()
