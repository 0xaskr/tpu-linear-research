import os
import sys
import torch
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Set Triton environment variables for CPU execution
os.environ["TRITON_CPU_BACKEND"] = "1"
os.environ["TRITON_INTERPRET"] = "1"

# Add local paths
sys.path.append(os.getcwd())
if os.path.exists(os.path.join(os.getcwd(), 'fla')):
    sys.path.append(os.path.join(os.getcwd(), 'fla'))

# Import JAX implementation
from attention_kda import ShortConvolution as JaxShortConvolution
from attention_kda import chunk_kda_reference

# Import PyTorch/Triton implementation
try:
    import fla.utils
    import contextlib
    
    # Patch utils
    fla.utils.custom_device_ctx = lambda index: contextlib.nullcontext()
    fla.utils.get_multiprocessor_count = lambda tensor_idx=0: 1
    
    # Import convolution module explicitly to patch its local reference
    from fla.modules import convolution
    convolution.get_multiprocessor_count = lambda tensor_idx=0: 1
    
    from fla.modules import ShortConvolution as PtShortConvolution
    from fla.ops.kda import chunk_kda as triton_chunk_kda
    HAS_FLA = True
except ImportError as e:
    print(f"Could not import FLA: {e}")
    HAS_FLA = False
    

def compare(name, pt_val, jax_val, atol=1e-4, rtol=1e-4):
    pt_np = pt_val.detach().cpu().float().numpy()
    jax_np = np.array(jax_val, dtype=np.float32)
    
    # Squeeze if needed (e.g. batch dim 1)
    if pt_np.shape != jax_np.shape:
        if pt_np.squeeze().shape == jax_np.squeeze().shape:
            pt_np = pt_np.squeeze()
            jax_np = jax_np.squeeze()
            
    diff = np.abs(pt_np - jax_np)
    max_diff = np.max(diff)
    print(f"[{name}] Max Diff: {max_diff:.6e}")
    if not np.allclose(pt_np, jax_np, atol=atol, rtol=rtol):
        print(f"[{name}] FAILED")
        print(f"  PT shape: {pt_np.shape}, JAX shape: {jax_np.shape}")
        # print first mismatch
        mismatch_idx = np.where(diff > atol + rtol * np.abs(jax_np))
        if len(mismatch_idx[0]) > 0:
             idx = tuple(x[0] for x in mismatch_idx)
             print(f"  First mismatch at {idx}: PT={pt_np[idx]}, JAX={jax_np[idx]}")
    else:
        print(f"[{name}] PASSED")

def short_conv_ref_pt(x, weight, bias=None, activation=None, cu_seqlens=None):
    """
    Reference implementation of ShortConvolution using standard PyTorch ops.
    x: [B, T, D] (B=1 if cu_seqlens is not None)
    weight: [D, 1, K]
    bias: [D]
    """
    B, T, D = x.shape
    K = weight.shape[2]
    padding = K - 1
    
    if cu_seqlens is None:
        # Standard padded [B, T, D]
        # Permute to [B, D, T]
        x_p = x.permute(0, 2, 1)
        x_pad = torch.nn.functional.pad(x_p, (padding, 0))
        out = torch.nn.functional.conv1d(x_pad, weight, bias=bias, groups=D)
        if activation == 'silu':
            out = torch.nn.functional.silu(out)
        return out.permute(0, 2, 1)
    
    # VarLen
    out_parts = []
    # x is [1, T, D] -> remove batch
    x_flat = x.squeeze(0)
    
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i+1]
        seq = x_flat[start:end] # [L, D]
        
        # [1, D, L]
        inp = seq.permute(1, 0).unsqueeze(0)
        inp_pad = torch.nn.functional.pad(inp, (padding, 0))
        
        # Conv
        out = torch.nn.functional.conv1d(inp_pad, weight, bias=bias, groups=D)
        
        if activation == 'silu':
            out = torch.nn.functional.silu(out)
            
        # [1, D, L] -> [L, D]
        out_parts.append(out.squeeze(0).permute(1, 0))
        
    res = torch.cat(out_parts, dim=0).unsqueeze(0)
    return res

def test_short_conv():
    print("\n=== Testing ShortConvolution (VarLen) ===")
    
    B, D, K = 1, 32, 4
    # VarLen: 2 sequences of length 10 and 20. Total = 30.
    seqlens = [10, 20]
    total_tokens = sum(seqlens)
    cu_seqlens = torch.tensor([0] + list(np.cumsum(seqlens)), dtype=torch.int32)
    
    # Inputs
    x = torch.randn(B, total_tokens, D, dtype=torch.float32) # [1, T, D]
    
    # Setup PyTorch weights (mimic FLA)
    # FLA weight: [D, 1, K]
    pt_w = torch.randn(D, 1, K, dtype=torch.float32)
    
    # Run Reference PyTorch
    out_pt = short_conv_ref_pt(x, pt_w, activation='silu', cu_seqlens=cu_seqlens)
    
    # Setup JAX layer
    rngs = nnx.Rngs(0)
    jax_conv = JaxShortConvolution(hidden_size=D, kernel_size=K, dtype=jnp.float32, precision="default", rngs=rngs)
    
    # Sync weights
    # PT: [D, 1, K]
    # JAX: [K, 1, D] (feature_group_count=D)
    # Transpose PT (D, 1, K) -> (K, 1, D)
    jax_w_val = jnp.array(pt_w.detach().numpy().transpose(2, 1, 0))
    jax_conv.conv.kernel.value = jax_w_val
    
    # Run JAX
    x_jax = jnp.array(x.numpy())
    cu_seqlens_jax = jnp.array(cu_seqlens.numpy())
    out_jax, _ = jax_conv(x_jax, cu_seqlens=cu_seqlens_jax)
    
    compare("ShortConv Output", out_pt, out_jax)

def kda_ref_pt(q, k, v, g, beta, cu_seqlens=None, initial_state=None):
    """
    Pure PyTorch reference for KDA kernel logic.
    q, k, v, g: [B, T, H, D]
    beta: [B, T, H]
    """
    B, T, H, K_dim = q.shape
    V_dim = v.shape[-1]
    
    if cu_seqlens is None:
        # Standard batch loop
        out = torch.zeros_like(v)
        final_states = []
        for b in range(B):
            state = torch.zeros(H, K_dim, V_dim, device=q.device)
            if initial_state is not None:
                state = state + initial_state[b]
            for t in range(T):
                # Update
                # state = state * exp(g) + beta * k * (v - k*state)
                # Actually Delta Rule: state = state * exp(g) + beta * k @ (v - k @ state)
                # Head loop for simplicity
                for h in range(H):
                    gh = g[b, t, h] # [D]
                    kh = k[b, t, h] # [D]
                    vh = v[b, t, h] # [V]
                    bh = beta[b, t, h]
                    
                    # Forget
                    state[h] = state[h] * torch.exp(gh)[:, None]
                    # Error
                    err = vh - torch.matmul(kh, state[h])
                    # Update
                    state[h] = state[h] + bh * torch.outer(kh, err)
                    # Output
                    out[b, t, h] = torch.matmul(q[b, t, h], state[h])
            final_states.append(state)
        return out, torch.stack(final_states)

    # VarLen
    out = torch.zeros_like(v)
    q_flat = q.squeeze(0)
    k_flat = k.squeeze(0)
    v_flat = v.squeeze(0)
    g_flat = g.squeeze(0)
    beta_flat = beta.squeeze(0)
    
    final_states = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i+1]
        state = torch.zeros(H, K_dim, V_dim, device=q.device)
        if initial_state is not None:
            state = state + initial_state[i]
            
        for t in range(start, end):
            for h in range(H):
                gh = g_flat[t, h]
                kh = k_flat[t, h]
                vh = v_flat[t, h]
                bh = beta_flat[t, h]
                
                state[h] = state[h] * torch.exp(gh)[:, None]
                err = vh - torch.matmul(kh, state[h])
                state[h] = state[h] + bh * torch.outer(kh, err)
                out[0, t, h] = torch.matmul(q_flat[t, h], state[h])
        final_states.append(state)
    return out, torch.stack(final_states)

def test_chunk_kda_kernel():
    print("\n=== Testing Chunk KDA Kernel (VarLen) ===")
    
    B, H, K_dim, V_dim = 1, 4, 16, 16
    seqlens = [8, 16]
    total_tokens = sum(seqlens)
    cu_seqlens = torch.tensor([0] + list(np.cumsum(seqlens)), dtype=torch.int32)
    
    # Inputs
    q = torch.randn(B, total_tokens, H, K_dim)
    k = torch.randn(B, total_tokens, H, K_dim)
    v = torch.randn(B, total_tokens, H, V_dim)
    # Use smaller values for g to avoid numerical instability in the recurrent loop
    g = -torch.rand(B, total_tokens, H, K_dim) * 0.1 
    beta = torch.rand(B, total_tokens, H) * 0.1
    
    # Run PyTorch Reference
    # Apply scale to q in PT ref to match JAX internal logic
    scale = K_dim ** -0.5
    out_pt, state_pt = kda_ref_pt(q * scale, k, v, g, beta, cu_seqlens=cu_seqlens)
    
    # Run JAX
    q_jax = jnp.array(q.numpy())
    k_jax = jnp.array(k.numpy())
    v_jax = jnp.array(v.numpy())
    g_jax = jnp.array(g.numpy())
    beta_jax = jnp.array(beta.numpy())
    cu_seqlens_jax = jnp.array(cu_seqlens.numpy())
    
    out_jax, state_jax = chunk_kda_reference(
        q=q_jax, k=k_jax, v=v_jax, g=g_jax, beta=beta_jax,
        cu_seqlens=cu_seqlens_jax,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False # Disable for direct logic comparison
    )
    
    compare("KDA Output (o)", out_pt, out_jax)
    compare("KDA Final State", state_pt, state_jax)

if __name__ == "__main__":
    test_short_conv()
    test_chunk_kda_kernel()

