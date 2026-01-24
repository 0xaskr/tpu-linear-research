
import os
import sys
import torch
import jax
import jax.numpy as jnp
import numpy as np

# Add the 'fla' directory to sys.path
fla_path = os.path.abspath(os.path.join(os.getcwd(), 'fla'))
sys.path.append(fla_path)

from pallas_kda_kernel import chunk_gated_delta_rule_fwd_h_jax

# Set environment variables for Triton CPU if needed
os.environ["TRITON_CPU_BACKEND"] = "1"
os.environ["TRITON_INTERPRET"] = "1"

try:
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    print("Successfully imported Triton reference")
except ImportError:
    print("Failed to import Triton reference, check fla path")
    import sys
    sys.exit(1)

def jax_reference(k, w, u, g=None, gk=None, initial_state=None, use_exp2=False, chunk_size=64):
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT
    
    h = jnp.zeros((B, NT, H, K, V), dtype=jnp.float32)
    v_new = jnp.zeros((B, T, H, V), dtype=jnp.float32)
    
    b_h = jnp.zeros((B, H, K, V), dtype=jnp.float32)
    if initial_state is not None:
        b_h = initial_state.astype(jnp.float32)
        
    for i_t in range(NT):
        h = h.at[:, i_t].set(b_h)
        
        start_t = i_t * BT
        curr_BT = min(BT, T - start_t)
        
        # Load chunk data
        b_k = k[:, start_t:start_t+BT, :, :].astype(jnp.float32)
        b_w = w[:, start_t:start_t+BT, :, :].astype(jnp.float32)
        b_u = u[:, start_t:start_t+BT, :, :].astype(jnp.float32)
        
        # v_pred = w @ h
        # b_w: [B, BT, H, K], b_h: [B, H, K, V]
        # output: [B, BT, H, V]
        b_v_pred = jnp.einsum('bthk,bhkv->bthv', b_w, b_h)
        
        # v_new = u - v_pred
        b_v_new = b_u - b_v_pred
        v_new = v_new.at[:, start_t:start_t+BT].set(b_v_new)
        
        if g is not None:
            b_g = g[:, start_t:start_t+BT, :].astype(jnp.float32)
            # last_idx = start_t + curr_BT - 1
            b_g_last = b_g[:, curr_BT-1, :] # [B, H]
            
            if use_exp2:
                b_decay_v = jnp.exp2(b_g_last[:, None, :] - b_g)
                b_decay_h = jnp.exp2(b_g_last)
            else:
                b_decay_v = jnp.exp(b_g_last[:, None, :] - b_g)
                b_decay_h = jnp.exp(b_g_last)
            
            # Mask out-of-bound
            mask = jnp.arange(BT) < curr_BT
            b_decay_v = jnp.where(mask[None, :, None], b_decay_v, 0.0)
            
            b_v_new = b_v_new * b_decay_v[:, :, :, None]
            b_h = b_h * b_decay_h[:, :, None, None]
            
        if gk is not None:
            b_gk = gk[:, start_t:start_t+BT, :, :].astype(jnp.float32)
            b_gk_last = b_gk[:, curr_BT-1, :, :] # [B, H, K]
            if use_exp2:
                b_h = b_h * jnp.exp2(b_gk_last[:, :, :, None])
            else:
                b_h = b_h * jnp.exp(b_gk_last[:, :, :, None])
                
        # h = h + k.T @ v_new
        # b_k: [B, BT, H, K], b_v_new: [B, BT, H, V]
        # output: [B, H, K, V]
        b_h = b_h + jnp.einsum('bthk,bthv->bhkv', b_k, b_v_new)
        
    return h, v_new, b_h

def test_accuracy():
    B, T, H, K, V = 2, 128, 4, 32, 64
    chunk_size = 64
    use_exp2 = True
    device = 'cpu'
    
    # Initialize inputs with small random values to avoid divergence
    torch.manual_seed(42)
    k_torch = torch.randn(B, T, H, K).to(device) * 0.1
    w_torch = torch.randn(B, T, H, K).to(device) * 0.1
    u_torch = torch.randn(B, T, H, V).to(device) * 0.1
    g_torch = torch.randn(B, T, H).to(device) * 0.1
    gk_torch = torch.randn(B, T, H, K).to(device) * 0.1
    h0_torch = torch.randn(B, H, K, V).to(device) * 0.1
    
    # Run Triton
    print("Running Triton reference...")
    h_tri, v_new_tri, ht_tri = chunk_gated_delta_rule_fwd_h(
        k=k_torch,
        w=w_torch,
        u=u_torch,
        g=g_torch,
        gk=gk_torch,
        initial_state=h0_torch,
        output_final_state=True,
        chunk_size=chunk_size,
        use_exp2=use_exp2
    )
    
    # Prepare JAX inputs
    k_jax = jnp.array(k_torch.numpy())
    w_jax = jnp.array(w_torch.numpy())
    u_jax = jnp.array(u_torch.numpy())
    g_jax = jnp.array(g_torch.numpy())
    gk_jax = jnp.array(gk_torch.numpy())
    h0_jax = jnp.array(h0_torch.numpy()) if h0_torch is not None else None
    
    # Run JAX Reference
    print("Running JAX reference...")
    h_ref, v_new_ref, ht_ref = jax_reference(
        k_jax, w_jax, u_jax, g_jax, gk_jax, h0_jax, use_exp2=use_exp2, chunk_size=chunk_size
    )
    
    # Run JAX Pallas
    print("Running JAX Pallas (Interpreter mode for CPU verification)...")
    # For CPU verification, we use interpret=True internally if needed, 
    # but here we just call the function which now has it OFF for hardware use.
    # To run on CPU, we can temporarily wrap it or set JAX to interpret mode.
    from jax.experimental import pallas as pl
    h_pal, v_new_pal, ht_pal = chunk_gated_delta_rule_fwd_h_jax(
        k_jax, w_jax, u_jax, g_jax, gk_jax, h0_jax,
        output_final_state=True, chunk_size=chunk_size, use_exp2=use_exp2
    )
    
    def check(name, a, b, tol=5e-3):
        if hasattr(a, 'numpy'): a = a.detach().cpu().numpy()
        if hasattr(b, 'numpy'): b = b.detach().cpu().numpy()
        a = np.array(a)
        b = np.array(b)
        diff = np.abs(a - b)
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"{name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
        return max_diff < tol

    print("\nComparing Pallas vs Triton:")
    s1 = check("h (history)", h_pal, h_tri)
    s2 = check("v_new", v_new_pal, v_new_tri)
    s3 = check("final_state", ht_pal, ht_tri)
    
    print("\nComparing Pallas vs JAX Reference:")
    check("h (history)", h_pal, h_ref)
    check("v_new", v_new_pal, v_new_ref)
    check("final_state", ht_pal, ht_ref)

    if s1 and s2 and s3:
        print("\nSUCCESS: Pallas kernel matches Triton reference!")
    else:
        print("\nFAILURE: Significant difference detected.")

if __name__ == "__main__":
    test_accuracy()
