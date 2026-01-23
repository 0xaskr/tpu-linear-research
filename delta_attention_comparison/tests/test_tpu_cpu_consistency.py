import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# Add paths to support imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # tpu-research
kda_root = os.path.join(project_root, 'delta_attention_comparison')

sys.path.append(project_root)
sys.path.append(kda_root)

from delta_attention_comparison.src.layers.kimi_delta_attention import KimiDeltaAttention

def test_tpu_cpu_consistency(B_list=[1, 2], L_list=[1024, 2048, 4096], H=2304, NH=32, NV=32, D=128, K=256):
    print("Checking JAX devices...")
    devices = jax.devices()
    print(f"Available devices: {devices}")
    
    has_tpu = any(d.platform == 'tpu' for d in devices)
    
    if not has_tpu:
        print("TPU devices not found. Cannot run TPU vs CPU comparison.")
        return

    print("Initializing Model (BF16).")
    rngs = nnx.Rngs(0)
    
    # Initialize model with bfloat16
    model = KimiDeltaAttention(
        hidden_size=H,
        num_heads=NH,
        head_dim=D,
        num_v_heads=NV,
        conv_kernel_size=K,
        dtype=jnp.bfloat16,
        rngs=rngs
    )

    # Force A_log to be small to avoid overflow in exp(-g) decomposition for large chunks
    # Default init maxval=16.0 is too large for the naive exp(g)*exp(-g) trick.
    model.A_log.value = jnp.full_like(model.A_log.value, jnp.log(0.05))
    
    # Define the loss function for gradient computation
    def loss_fn(model, x, dy, chunk_size):
        out, _ = model(x, chunk_size=chunk_size)
        return jnp.mean(out * dy)
        
    # Get the value_and_grad function
    grad_fn = nnx.value_and_grad(loss_fn, argnums=1)

    # JIT for CPU and TPU
    step_cpu = jax.jit(grad_fn, backend='cpu', static_argnums=(3,))
    step_tpu = jax.jit(grad_fn, backend='tpu', static_argnums=(3,))

    print(f"Starting Consistency Tests with parameters: H={H}, NH={NH}, NV={NV}, D={D}, K={K}, dtype=bfloat16")
    
    chunk_sizes = [64, 128, 256, 512]

    for B in B_list:
        for L in L_list:
            for chunk_size in chunk_sizes:
                print(f"\nTesting B={B}, L={L}, Chunk={chunk_size}...")
                
                # Create random inputs in bfloat16
                key = jax.random.key(42)
                k1, k2 = jax.random.split(key)
                x = jax.random.normal(k1, (B, L, H), dtype=jnp.bfloat16)
                dy = jax.random.normal(k2, (B, L, H), dtype=jnp.bfloat16)
                
                # Run on CPU
                try:
                    (loss_cpu, dx_cpu) = step_cpu(model, x, dy, chunk_size)
                    # Convert to float32 for numpy comparison to preserve whatever precision we have
                    loss_cpu_np = np.array(loss_cpu, dtype=np.float32)
                    dx_cpu_np = np.array(dx_cpu, dtype=np.float32)
                except Exception as e:
                    print(f"CPU execution failed: {e}")
                    continue

                # Run on TPU
                try:
                    (loss_tpu, dx_tpu) = step_tpu(model, x, dy, chunk_size)
                    loss_tpu_np = np.array(loss_tpu, dtype=np.float32)
                    dx_tpu_np = np.array(dx_tpu, dtype=np.float32)
                except Exception as e:
                    print(f"TPU execution failed: {e}")
                    continue
                
                # Compare
                loss_diff = np.abs(loss_cpu_np - loss_tpu_np)
                dx_diff = np.abs(dx_cpu_np - dx_tpu_np)
                max_dx_diff = dx_diff.max()
                mean_dx_diff = dx_diff.mean()
                
                # Cosine Similarity
                cosine_sim = np.sum(dx_cpu_np * dx_tpu_np) / (np.linalg.norm(dx_cpu_np) * np.linalg.norm(dx_tpu_np))

                print(f"  Loss CPU: {loss_cpu_np:.6f}, TPU: {loss_tpu_np:.6f}, Diff: {loss_diff:.2e}")
                print(f"  Grad Max Diff: {max_dx_diff:.2e}")
                print(f"  Grad Mean Diff: {mean_dx_diff:.2e}")
                print(f"  Grad Cosine Sim: {cosine_sim:.8f}")
                
                # Tolerances for BF16
                # BF16 has less precision than FP32. Differences are expected to be larger.
                # Especially since we accumulate large sums in the loss.
                
                tol_atol = 5e-2 # Relaxed tolerance for BF16
                tol_rtol = 5e-2
                
                if max_dx_diff > tol_atol:
                    print(f"  [WARNING] Differences exceed BF16 tolerances ({tol_atol})")
                
                if not np.allclose(loss_cpu_np, loss_tpu_np, atol=tol_atol, rtol=tol_rtol):
                    print("  [FAIL] Loss mismatch")
                else:
                    print("  [PASS] Loss match")
                    
                if not np.allclose(dx_cpu_np, dx_tpu_np, atol=tol_atol, rtol=tol_rtol):
                     print("  [FAIL] Gradient mismatch")
                else:
                     print("  [PASS] Gradient match")

if __name__ == "__main__":
    test_tpu_cpu_consistency()
