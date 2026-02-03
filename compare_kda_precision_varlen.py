import os
os.environ["TRITON_CPU_BACKEND"] = "1"
os.environ["TRITON_INTERPRET"] = "1"
import torch
import jax
import jax.numpy as jnp
import numpy as np
import sys

# Add local directory to path
sys.path.append(os.getcwd())
if os.path.exists(os.path.join(os.getcwd(), 'fla')):
    sys.path.append(os.path.join(os.getcwd(), 'fla'))

# Attempt imports
try:
    from test_pallas_manual_varlen import chunk_gated_delta_rule_fwd_h_varlen as pallas_fwd
    from test_pallas_manual_varlen import prepare_chunk_indices
    print("Successfully imported Pallas kernel wrapper.")
except ImportError as e:
    print(f"Failed to import Pallas kernel wrapper: {e}")
    pallas_fwd = None

try:
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as triton_fwd
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton_fwd = None
    print("Triton kernel not found.")

def compare_tensor(name, pt_t, jax_t, atol=1e-5, rtol=1e-5):
    if pt_t is None and jax_t is None:
        print(f"[{name}] Both are None. MATCH.")
        return
    if pt_t is None or jax_t is None:
        print(f"[{name}] One is None! MISMATCH.")
        return

    pt_val = pt_t.detach().cpu().float().numpy() if isinstance(pt_t, torch.Tensor) else np.array(pt_t)
    jax_val = np.array(jax_t.astype(jnp.float32)) if hasattr(jax_t, 'dtype') and jax_t.dtype == jnp.bfloat16 else np.array(jax_t)

    if pt_val.shape != jax_val.shape:
        print(f"[{name}] Shape mismatch: Left {pt_val.shape} vs Right {jax_val.shape}. FAIL.")
        if pt_val.squeeze().shape == jax_val.squeeze().shape:
            print(f"  Attempting comparison with squeezed shapes: {pt_val.squeeze().shape}")
            pt_val = pt_val.squeeze()
            jax_val = jax_val.squeeze()
        else:
            return

    diff = np.abs(pt_val - jax_val)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    tolerance = atol + rtol * np.abs(jax_val)
    error_ratio = diff / (tolerance + 1e-12)
    max_error_ratio = np.max(error_ratio)

    is_close = np.allclose(pt_val, jax_val, atol=atol, rtol=rtol)
    status = "PASS" if is_close else "FAIL"

    print(f"[{name}] {status}")
    print(f"  Max Abs Diff     : {max_diff:.6e}")
    print(f"  Max Error Ratio  : {max_error_ratio:.6f} (<= 1.0 is Pass)")
    print(f"  Mean Diff        : {mean_diff:.6e}")

    if not is_close:
        idx = np.unravel_index(np.argmax(error_ratio), error_ratio.shape)
        print(f"  Max Mismatch details at index {idx}:")
        print(f"    Left (Triton)  = {pt_val[idx]}")
        print(f"    Right (Pallas) = {jax_val[idx]}")
        print(f"    Diff           = {diff[idx]}")
        print(f"    Tolerance      = {tolerance[idx]} (atol={atol} + rtol={rtol}*|Right|)")
        print(f"    Ratio          = {error_ratio[idx]}")

def run_comparison_varlen_fp32():
    print("\n" + "="*40)
    print("Running Varlen Comparison FP32 (chunk_gated_delta_rule_fwd_h)")
    print("="*40)

    # Configuration
    rng_dtype = torch.bfloat16
    triton_dtype = torch.float32
    jax_dtype = jnp.bfloat16

    seqlens_list = [64, 128, 64]
    N = len(seqlens_list)
    TotalT = sum(seqlens_list)
    chunk_size = 64
    B, H, K, V = 1, 4, 64, 64

    print(f"N={N}, Seqlens={seqlens_list}, TotalT={TotalT}, H={H}, K={K}, V={V}, chunk_size={chunk_size}")

    torch.manual_seed(42)
    k = torch.randn((B, TotalT, H, K), dtype=rng_dtype)
    w = torch.randn((B, TotalT, H, K), dtype=rng_dtype)
    u = torch.randn((B, TotalT, H, V), dtype=rng_dtype)
    h0 = torch.randn((N, H, K, V), dtype=rng_dtype)

    # Generate chunk-local cumulative log decay for g and gk
    def chunk_local_cumsum(x, chunk_size):
        out = x.clone()
        for i in range(0, x.shape[1], chunk_size):
            end = min(i + chunk_size, x.shape[1])
            out[:, i:end] = x[:, i:end].cumsum(1)
        return out

    raw_g = -torch.randn((B, TotalT, H), dtype=rng_dtype).abs() * 0.01
    g = chunk_local_cumsum(raw_g, chunk_size)

    raw_gk = -torch.randn((B, TotalT, H, K), dtype=rng_dtype).abs() * 0.01
    gk = chunk_local_cumsum(raw_gk, chunk_size)

    cu_seqlens = torch.tensor([0] + list(np.cumsum(seqlens_list)), dtype=torch.int32)

    k_pt = torch.tensor(k, device="cpu", dtype=triton_dtype)
    w_pt = torch.tensor(w, device="cpu", dtype=triton_dtype)
    u_pt = torch.tensor(u, device="cpu", dtype=triton_dtype)
    g_pt = torch.tensor(g, device="cpu", dtype=triton_dtype)
    gk_pt = torch.tensor(gk, device="cpu", dtype=triton_dtype)
    h0_pt = torch.tensor(h0, device="cpu", dtype=triton_dtype)
    # Triton Run
    print("\nRunning Triton varlen (FP32)...")
    h_ref, v_new_ref, final_state_ref = triton_fwd(
        k=k_pt, w=w_pt, u=u_pt, g=g_pt, gk=gk_pt,
        initial_state=h0_pt, output_final_state=True,
        chunk_size=chunk_size, save_new_value=True,
        cu_seqlens=cu_seqlens.long(),
        use_exp2=False
    )
    print(f"Triton h_ref shape: {h_ref.shape}")

    # Pallas Run
    print("\nRunning Pallas varlen (FP32)...")
    k_jax = jnp.array(k.to(torch.float32), dtype=jax_dtype)
    w_jax = jnp.array(w.to(torch.float32), dtype=jax_dtype)
    u_jax = jnp.array(u.to(torch.float32), dtype=jax_dtype)
    g_jax = jnp.array(g.to(torch.float32), dtype=jax_dtype)
    gk_jax = jnp.array(gk.to(torch.float32), dtype=jax_dtype)
    h0_jax = jnp.array(h0.to(torch.float32), dtype=jax_dtype)
    cu_seqlens_jax = jnp.array(cu_seqlens.numpy(), dtype=jnp.int32)

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunk_indices_jax = jnp.array(chunk_indices.numpy(), dtype=jnp.int32)

    h_history_jax, v_new_jax, final_state_jax = pallas_fwd(
        k=k_jax, w=w_jax, u=u_jax, g=g_jax, gk=gk_jax,
        initial_state=h0_jax, output_final_state=True,
        chunk_size=chunk_size, save_new_value=True,
        seqlens=cu_seqlens_jax,
        chunk_indices=chunk_indices_jax,
        use_exp2=False
    )
    jax.block_until_ready(h_history_jax)
    print(f"Pallas h_jax shape: {h_history_jax.shape}")

    print("\n" + "="*40)
    print("COMPARISON RESULTS (FP32)")
    print("="*40)

    # Tolerances for FP32
    atol, rtol = 1e-3, 1e-4

    compare_tensor("Hidden States (h)", h_ref, h_history_jax, atol=atol, rtol=rtol)
    compare_tensor("Residual (v_new)", v_new_ref, v_new_jax, atol=atol, rtol=rtol)
    compare_tensor("Final State (ht)", final_state_ref, final_state_jax, atol=atol, rtol=rtol)

if __name__ == "__main__":
    run_comparison_varlen_fp32()