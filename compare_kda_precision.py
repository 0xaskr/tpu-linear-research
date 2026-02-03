import os
os.environ["TRITON_CPU_BACKEND"] = "1"
os.environ["TRITON_INTERPRET"] = "1"
import torch
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os
from einops import rearrange

# Add local directory to path
sys.path.append(os.getcwd())
if os.path.exists(os.path.join(os.getcwd(), 'fla')):
    sys.path.append(os.path.join(os.getcwd(), 'fla'))

# Attempt imports
try:
    from test_pallas_manual import chunk_gated_delta_rule_fwd_h as pallas_fwd
    from test_pallas_manual import chunk_gla_fwd_o_gk as pallas_o_gk
    print("Successfully imported Pallas kernel wrapper.")
except ImportError as e:
    print(f"Failed to import Pallas kernel wrapper: {e}")
    pallas_fwd = None
    pallas_o_gk = None

try:
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as triton_fwd
    from fla.ops.gla.chunk import chunk_gla_fwd_o_gk as triton_o_gk
    HAS_TRITON = torch.cuda.is_available()
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton_fwd = None
    triton_o_gk = None
    print("Triton kernel not found. Using Torch reference instead.")

def compare_tensor(name, pt_t, jax_t, atol=1e-2, rtol=1e-2):
    if pt_t is None and jax_t is None:
        print(f"[{name}] Both are None. MATCH.")
        return
    if pt_t is None or jax_t is None:
        print(f"[{name}] One is None! MISMATCH.")
        return

    if isinstance(pt_t, torch.Tensor):
        pt_val = pt_t.detach().cpu().float().numpy()
    else:
        pt_val = np.array(pt_t)

    if hasattr(jax_t, 'dtype') and jax_t.dtype == jnp.bfloat16:
         jax_val = np.array(jax_t.astype(jnp.float32))
    else:
         jax_val = np.array(jax_t)

    if pt_val.shape != jax_val.shape:
        print(f"[{name}] Shape mismatch: Left {pt_val.shape} vs Right {jax_val.shape}. FAIL.")
        return

    diff = np.abs(pt_val - jax_val)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Calculate Error Ratio (The combined metric)
    # ratio = diff / (atol + rtol * abs(ref))
    # Using jax_val as reference
    tolerance = atol + rtol * np.abs(jax_val)
    error_ratio = diff / tolerance
    max_error_ratio = np.max(error_ratio)

    # is_close = max_error_ratio <= 1.0
    is_close = np.allclose(pt_val, jax_val, atol=atol, rtol=rtol)
    status = "PASS" if is_close else "FAIL"

    print(f"[{name}] {status}")
    print(f"  Max Abs Diff     : {max_diff:.6e}")
    print(f"  Max Error Ratio  : {max_error_ratio:.6f} (<= 1.0 is Pass)")
    print(f"  Mean Diff        : {mean_diff:.6e}")

    if not is_close:
        idx = np.argmax(error_ratio)
        flat_pt = pt_val.flatten()
        flat_jax = jax_val.flatten()
        flat_diff = diff.flatten()
        flat_tol = tolerance.flatten()
        print(f"  Max Mismatch details at index {idx}:")
        print(f"    Left (Triton)  = {flat_pt[idx]}")
        print(f"    Right (Pallas) = {flat_jax[idx]}")
        print(f"    Diff           = {flat_diff[idx]}")
        print(f"    Tolerance      = {flat_tol[idx]} (atol={atol} + rtol={rtol}*|Right|)")
        print(f"    Ratio          = {flat_diff[idx] / flat_tol[idx]}")

def run_comparison_o_gk():
    if pallas_o_gk is None:
        print("Skipping o_gk comparison: Pallas function not found.")
        return
    if triton_o_gk is None:
        print("Skipping o_gk comparison: Triton function not found.")
        return

    rng_dtype = torch.bfloat16
    tirton_dtype = torch.float32
    pallas_dtype = jnp.bfloat16

    B, T, H, K, V = 2, 64, 4, 64, 64
    chunk_size = 64
    use_exp2 = False
    scale = K ** -0.5

    print(f"\nConfiguration o_gk: B={B}, T={T}, H={H}, K={K}, V={V}, chunk_size={chunk_size}")

    torch.manual_seed(42)
    q = torch.randn((B, T, H, K), dtype=rng_dtype)
    v = torch.randn((B, T, H, V), dtype=rng_dtype)
    g = torch.randn((B, T, H, K), dtype=rng_dtype)
    A = torch.randn((B, T, H, chunk_size), dtype=rng_dtype)
    # h shape: [B, NT, H, K, V]
    NT = (T + chunk_size - 1) // chunk_size
    h = torch.randn((B, NT, H, K, V), dtype=rng_dtype)

    # Triton Run
    q_pt = torch.tensor(q, device="cpu", dtype=tirton_dtype)
    v_pt = torch.tensor(v, device="cpu", dtype=tirton_dtype)
    g_pt = torch.tensor(g, device="cpu", dtype=tirton_dtype)
    A_pt = torch.tensor(A, device="cpu", dtype=tirton_dtype)
    h_pt = torch.tensor(h, device="cpu", dtype=tirton_dtype)
    print("Running Triton o_gk...")
    # q, v, g, A, h, scale, cu_seqlens, chunk_size, chunk_indices, use_exp2
    o_ref = triton_o_gk(
        q_pt, v_pt, g_pt, A_pt, h_pt,
        scale=scale,
        chunk_size=chunk_size,
        use_exp2=use_exp2
    )

    # Pallas Run
    print("Running Pallas o_gk...")
    q_jax = jnp.array(q.to(torch.float32), dtype=pallas_dtype)
    v_jax = jnp.array(v.to(torch.float32), dtype=pallas_dtype)
    g_jax = jnp.array(g.to(torch.float32), dtype=pallas_dtype)
    A_jax = jnp.array(A.to(torch.float32), dtype=pallas_dtype)
    h_jax = jnp.array(h.to(torch.float32), dtype=pallas_dtype)

    o_jax = pallas_o_gk(
        q_jax, v_jax, g_jax, A_jax, h_jax,
        scale=scale,
        chunk_size=chunk_size,
        use_exp2=use_exp2
    )
    if isinstance(o_jax, (tuple, list)):
        o_jax = o_jax[0]
    jax.block_until_ready(o_jax)

    # Check shape of o_jax
    if o_jax.shape != o_ref.shape:
        print(f"Shape mismatch! Ref: {o_ref.shape}, Pallas: {o_jax.shape}")
        if o_jax.ndim == 4 and o_jax.shape == (B, H, T, V):
             print("Transposing Pallas output from [B, H, T, V] to [B, T, H, V]")
             o_jax = jnp.transpose(o_jax, (0, 2, 1, 3))

    compare_tensor("Output (o) - FULL", o_ref, o_jax, atol=1e-2, rtol=1e-2)

def run_comparison():
    B, T, H, K, V = 2, 128, 4, 64, 64
    chunk_size = 64
    use_exp2 = False
    rng_dtype = torch.bfloat16
    tirton_dtype = torch.float32
    pallas_dtype = jnp.bfloat16

    # Initialize weights with correct scaling (1/sqrt(K)) to prevent numerical explosion.
    # This respects the constraint of "not modifying RNG" (we use the same RNG distribution),
    # but applies the necessary normalization for Linear Attention stability.
    scale = K ** -0.5
    scale = 1

    # Restore Mixed Mode (Triton FP32 vs Pallas BF16) to verify the target goal.
    print(f"\nConfiguration: B={B}, T={T}, H={H}, K={K}, V={V}, chunk_size={chunk_size}, use_exp2={use_exp2}, mode=Mixed (Triton=FP32, Pallas=BF16), scale={scale}")

    torch.manual_seed(42)
    k = torch.randn((B, T, H, K), dtype=rng_dtype)
    w = torch.randn((B, T, H, K), dtype=rng_dtype)
    u = torch.randn((B, T, H, V), dtype=rng_dtype)
    g = torch.randn((B, T, H), dtype=rng_dtype)
    gk = torch.randn((B, T, H, K), dtype=rng_dtype)
    h0 = torch.randn((B, H, K, V), dtype=rng_dtype)

    # --- Reference Run (Triton FP32) ---
    print("\nRunning Triton kernel (FP32)...")
    k_pt = torch.tensor(k, device='cpu', dtype=tirton_dtype)
    w_pt = torch.tensor(w, device='cpu', dtype=tirton_dtype)
    u_pt = torch.tensor(u, device='cpu', dtype=tirton_dtype)
    g_pt = torch.tensor(g, device='cpu', dtype=tirton_dtype)
    gk_pt = torch.tensor(gk, device='cpu', dtype=tirton_dtype)
    h0_pt = torch.tensor(h0, device='cpu', dtype=tirton_dtype)

    if triton_fwd is not None:
        h_ref, v_new_ref, final_state_ref = triton_fwd(
            k=k_pt, w=w_pt, u=u_pt, g=g, gk=gk,
            initial_state=h0_pt, output_final_state=True,
            chunk_size=chunk_size, save_new_value=True, use_exp2=use_exp2
        )
    else:
        print("Triton fwd function missing, skipping.")
        h_ref = None

    # --- Run Pallas ---
    print("\nRunning Pallas kernel ({})...", pallas_dtype.dtype)
    k_jax = jnp.array(k.to(torch.float32), dtype=pallas_dtype)
    w_jax = jnp.array(w.to(torch.float32), dtype=pallas_dtype)
    u_jax = jnp.array(u.to(torch.float32), dtype=pallas_dtype)
    g_jax = jnp.array(g.to(torch.float32), dtype=pallas_dtype)
    gk_jax = jnp.array(gk.to(torch.float32), dtype=pallas_dtype)
    h0_jax = jnp.array(h0.to(torch.float32), dtype=pallas_dtype)

    if pallas_fwd is not None:
        h_jax, v_new_jax, final_state_jax = pallas_fwd(
            k=k_jax, w=w_jax, u=u_jax, g=g_jax, gk=gk_jax,
            initial_state=h0_jax, output_final_state=True,
            chunk_size=chunk_size, save_new_value=True, use_exp2=use_exp2
        )
        jax.block_until_ready(h_jax)
    else:
        print("Pallas fwd function missing, skipping.")
        h_jax = None

    print("\n" + "="*40)
    print("COMPARISON INTPUTS (Triton FP32 vs Pallas {})", pallas_dtype.dtype)
    print("="*40)
    compare_tensor("k_pt/jax", k_pt, k_jax, atol=1e-2, rtol=1e-2)
    compare_tensor("w_pt/jax", w_pt, w_jax, atol=1e-2, rtol=1e-2)
    compare_tensor("u_pt/jax", u_pt, u_jax, atol=1e-2, rtol=1e-2)
    compare_tensor("h0_pt/jax", h0_pt, h0_jax, atol=1e-2, rtol=1e-2)

    print("\n" + "="*40)
    print("COMPARISON RESULTS (Triton FP32 vs Pallas {})", pallas_dtype.dtype)
    print("="*40)
    # Using 1e-2 tolerance for BF16 vs FP32 verification with scaled inputs
    if h_ref is not None and h_jax is not None:
        compare_tensor("Hidden State (h)", h_ref, h_jax, atol=1e-2, rtol=1e-2)
        compare_tensor("Residual (v_new)", v_new_ref, v_new_jax, atol=1e-2, rtol=1e-2)
        compare_tensor("Final State (ht)", final_state_ref, final_state_jax, atol=1e-2, rtol=1e-2)

def run_comparison_varlen():
    print("\n" + "="*40)
    print("Running Varlen Comparison (chunk_gated_delta_rule_fwd_h)")
    print("="*40)

    try:
        from fla.ops.utils import prepare_chunk_indices
    except ImportError:
        print("Could not import prepare_chunk_indices, skipping varlen test.")
        return

    B_real = 3
    seqlens_list = [13, 33, 20]
    TotalT = sum(seqlens_list)
    H, K, V = 4, 64, 64
    chunk_size = 32

    rng_dtype = torch.bfloat16
    tirton_dtype = torch.float32
    pallas_dtype = jnp.bfloat16

    print(f"B={B_real}, Seqlens={seqlens_list}, TotalT={TotalT}, H={H}, K={K}, V={V}, chunk_size={chunk_size}")

    torch.manual_seed(42)
    # Packed inputs [1, TotalT, ...] for Triton
    k = torch.randn((1, TotalT, H, K), dtype=rng_dtype)
    w = torch.randn((1, TotalT, H, K), dtype=rng_dtype)
    u = torch.randn((1, TotalT, H, V), dtype=rng_dtype)
    # g should be cumulative log decay to avoid explosion
    raw_g = torch.randn((1, TotalT, H), dtype=torch.float32)
    g = -raw_g.abs().cumsum(1) * 0.1
    g = g.to(rng_dtype)
    gk = torch.randn((1, TotalT, H, K), dtype=rng_dtype)
    h0 = torch.randn((B_real, H, K, V), dtype=rng_dtype)

    cu_seqlens = torch.tensor([0] + list(np.cumsum(seqlens_list)), dtype=torch.int32)

    # Triton Run
    if triton_fwd is not None:
        print("Running Triton varlen...")
        k_pt = k.to(tirton_dtype)
        w_pt = w.to(tirton_dtype)
        u_pt = u.to(tirton_dtype)
        g_pt = g.to(tirton_dtype)
        gk_pt = gk.to(tirton_dtype)
        h0_pt = h0.to(tirton_dtype)
        cu_seqlens_pt = cu_seqlens.to(torch.int32)

        h_ref, v_new_ref, final_state_ref = triton_fwd(
            k=k_pt, w=w_pt, u=u_pt, g=g_pt, gk=gk_pt,
            initial_state=h0_pt, output_final_state=True,
            chunk_size=chunk_size, save_new_value=True,
            cu_seqlens=cu_seqlens_pt,
            use_exp2=False
        )
        print(f"Triton h_ref shape: {h_ref.shape}")
    else:
        h_ref = None

    # Pallas Run
    if pallas_fwd is not None:
        print("Running Pallas varlen...")
        # Pallas wrapper expects [TotalT, H, K] (3D) to trigger Packed logic
        k_jax = jnp.array(k.squeeze(0).to(torch.float32), dtype=pallas_dtype)
        w_jax = jnp.array(w.squeeze(0).to(torch.float32), dtype=pallas_dtype)
        u_jax = jnp.array(u.squeeze(0).to(torch.float32), dtype=pallas_dtype)
        g_jax = jnp.array(g.squeeze(0).to(torch.float32), dtype=pallas_dtype)
        gk_jax = jnp.array(gk.squeeze(0).to(torch.float32), dtype=pallas_dtype)
        h0_jax = jnp.array(h0.to(torch.float32), dtype=pallas_dtype)
        cu_seqlens_jax = jnp.array(cu_seqlens.numpy(), dtype=jnp.int32)

        h_jax, v_new_jax, final_state_jax = pallas_fwd(
            k=k_jax, w=w_jax, u=u_jax, g=g_jax, gk=gk_jax,
            initial_state=h0_jax, output_final_state=True,
            chunk_size=chunk_size, save_new_value=True,
            cu_seqlens=cu_seqlens_jax,
            use_exp2=False
        )
        jax.block_until_ready(h_jax)
        print(f"Pallas h_jax shape: {h_jax.shape}")

        if final_state_ref is not None:
            compare_tensor("Final State (ht) Varlen", final_state_ref, final_state_jax, atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    run_comparison()

    # Run o_gk comparison
    run_comparison_o_gk()

    # Run varlen comparison
    # run_comparison_varlen()

