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
    from test_pallas_manual import kernel_wrapper as pallas_fwd
    print("Successfully imported Pallas kernel wrapper.")
except ImportError as e:
    print(f"Failed to import Pallas kernel wrapper: {e}")
    sys.exit(1)

try:
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as triton_fwd
    HAS_TRITON = torch.cuda.is_available()
    if HAS_TRITON:
        print("Successfully imported Triton kernel.")
    else:
        print("Triton kernel imported but CUDA not available. Using Torch reference instead.")
except ImportError:
    HAS_TRITON = False
    print("Triton kernel not found. Using Torch reference instead.")

def compare_tensor(name, pt_t, jax_t, atol=1e-4, rtol=1e-4):
    if pt_t is None and jax_t is None:
        print(f"[{name}] Both are None. MATCH.")
        return
    if pt_t is None or jax_t is None:
        print(f"[{name}] One is None! MISMATCH.")
        return

    if isinstance(pt_t, torch.Tensor):
        pt_val = pt_t.detach().cpu().numpy()
    else:
        pt_val = np.array(pt_t)
        
    jax_val = np.array(jax_t)

    if pt_val.shape != jax_val.shape:
        print(f"[{name}] Shape mismatch: Left {pt_val.shape} vs Right {jax_val.shape}. FAIL.")
        return

    diff = np.abs(pt_val - jax_val)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    is_close = np.allclose(pt_val, jax_val, atol=atol, rtol=rtol)
    status = "PASS" if is_close else "FAIL"
    
    print(f"[{name}] {status}")
    print(f"  Max Diff : {max_diff:.6e}")
    print(f"  Mean Diff: {mean_diff:.6e}")
    
    if not is_close:
        flat_pt = pt_val.flatten()
        flat_jax = jax_val.flatten()
        flat_diff = diff.flatten()
        idx = np.argmax(flat_diff)
        print(f"  Max mismatch at index {idx}: Left={flat_pt[idx]}, Right={flat_jax[idx]}")

def chunk_gated_delta_rule_fwd_h_torch(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT

    h = k.new_empty(B, NT, H, K, V, dtype=torch.float32)
    v_new = torch.empty_like(u) if save_new_value else None
    
    current_state = k.new_zeros(B, H, K, V, dtype=torch.float32)
    if initial_state is not None:
        current_state += initial_state.to(torch.float32)

    for i_t in range(NT):
        h[:, i_t] = current_state.clone()
        start, end = i_t * BT, min((i_t + 1) * BT, T)
        
        k_t = k[:, start:end]
        w_t = w[:, start:end]
        u_t = u[:, start:end]

        proj = torch.einsum('bthk,bhkv->bthv', w_t.to(torch.float32), current_state)
        v_new_t = u_t.to(torch.float32) - proj
        
        if save_new_value:
            v_new[:, start:end] = v_new_t.to(v_new.dtype)

        if g is not None or gk is not None:
            last_idx = end - 1
            if g is not None:
                g_t = g[:, start:end]
                g_last = g[:, last_idx]
                if use_exp2:
                    decay_v = torch.exp2(g_last.unsqueeze(1) - g_t)
                    decay_s = torch.exp2(g_last)
                else:
                    decay_v = torch.exp(g_last.unsqueeze(1) - g_t)
                    decay_s = torch.exp(g_last)
                v_new_t = v_new_t * decay_v.unsqueeze(-1)
                current_state = current_state * decay_s.unsqueeze(-1).unsqueeze(-1)
            if gk is not None:
                gk_last = gk[:, last_idx]
                if use_exp2:
                    decay_gk = torch.exp2(gk_last)
                else:
                    decay_gk = torch.exp(gk_last)
                current_state = current_state * decay_gk.unsqueeze(-1)

        update = torch.einsum('bthk,bthv->bhkv', k_t.to(torch.float32), v_new_t)
        current_state += update

    final_state = current_state if output_final_state else None
    return h.to(k.dtype), v_new, final_state

def run_comparison():
    B, T, H, K, V = 2, 128, 4, 64, 64
    chunk_size = 64
    use_exp2 = True
    dtype = np.float32
    
    print(f"\nConfiguration: B={B}, T={T}, H={H}, K={K}, V={V}, chunk_size={chunk_size}, use_exp2={use_exp2}")

    np.random.seed(42)
    k_np = (np.random.randn(B, T, H, K) * 0.1).astype(dtype)
    w_np = (np.random.randn(B, T, H, K) * 0.1).astype(dtype)
    u_np = (np.random.randn(B, T, H, V) * 0.1).astype(dtype)
    g_np = (np.random.randn(B, T, H) * 0.05 - 0.1).astype(dtype)
    gk_np = (np.random.randn(B, T, H, K) * 0.05 - 0.1).astype(dtype)
    h0_np = (np.random.randn(B, H, K, V) * 0.1).astype(dtype)

    # --- Reference Run ---
    if HAS_TRITON:
        print("\nRunning Triton kernel...")
        k_pt = torch.tensor(k_np, device='cuda')
        w_pt = torch.tensor(w_np, device='cuda')
        u_pt = torch.tensor(u_np, device='cuda')
        g_pt = torch.tensor(g_np, device='cuda')
        gk_pt = torch.tensor(gk_np, device='cuda')
        h0_pt = torch.tensor(h0_np, device='cuda')
        h_ref, v_new_ref, final_state_ref = triton_fwd(
            k=k_pt, w=w_pt, u=u_pt, g=g_pt, gk=gk_pt,
            initial_state=h0_pt, output_final_state=True,
            chunk_size=chunk_size, save_new_value=True, use_exp2=use_exp2
        )
    else:
        print("\nRunning Torch reference...")
        k_pt = torch.tensor(k_np)
        w_pt = torch.tensor(w_np)
        u_pt = torch.tensor(u_np)
        g_pt = torch.tensor(g_np)
        gk_pt = torch.tensor(gk_np)
        h0_pt = torch.tensor(h0_np)
        h_ref, v_new_ref, final_state_ref = chunk_gated_delta_rule_fwd_h_torch(
            k=k_pt, w=w_pt, u=u_pt, g=g_pt, gk=gk_pt,
            initial_state=h0_pt, output_final_state=True,
            chunk_size=chunk_size, save_new_value=True, use_exp2=use_exp2
        )

    # --- Run Pallas ---
    print("\nRunning Pallas kernel...")
    k_jax = jnp.array(k_np)
    w_jax = jnp.array(w_np)
    u_jax = jnp.array(u_np)
    g_jax = jnp.array(g_np)
    gk_jax = jnp.array(gk_np)
    h0_jax = jnp.array(h0_np)

    h_jax, v_new_jax, final_state_jax = pallas_fwd(
        k=k_jax, w=w_jax, u=u_jax, g=g_jax, gk=gk_jax,
        initial_state=h0_jax, output_final_state=True,
        chunk_size=chunk_size, save_new_value=True, use_exp2=use_exp2
    )
    jax.block_until_ready(h_jax)

    print("\n" + "="*40)
    print("COMPARISON RESULTS")
    print("="*40)
    compare_tensor("Hidden State (h)", h_ref, h_jax, atol=2e-3, rtol=2e-3)
    compare_tensor("Residual (v_new)", v_new_ref, v_new_jax, atol=2e-3, rtol=2e-3)
    compare_tensor("Final State (ht)", final_state_ref, final_state_jax, atol=5e-3, rtol=5e-3)

if __name__ == "__main__":
    run_comparison()
