
import os
import sys
import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax import nnx

# Add paths to sys.path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # tpu-research
kda_root = os.path.join(project_root, 'delta_attention_comparison')
fla_root = os.path.join(project_root, 'fla')

if project_root not in sys.path:
    sys.path.append(project_root)
if kda_root not in sys.path:
    sys.path.append(kda_root)
if fla_root not in sys.path:
    sys.path.append(fla_root)

# Import Models
try:
    from delta_attention_comparison.src.layers.kimi_delta_attention import KimiDeltaAttention as JaxKDA
    from fla.layers.kda import KimiDeltaAttention as TorchKDA
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)

def t2j(t):
    """Convert PyTorch tensor to JAX array, ensuring float32"""
    return jnp.array(t.detach().cpu().float().numpy())

def set_linear(jax_linear, pt_linear, transpose=True):
    """Copy weights from PyTorch Linear to JAX DenseGeneral/Linear"""
    w = pt_linear.weight.detach().cpu().float()
    if transpose:
        w = w.T
    jax_linear.kernel.value = jnp.array(w.numpy())
    
    if hasattr(pt_linear, 'bias') and pt_linear.bias is not None:
        if hasattr(jax_linear, 'bias') and jax_linear.bias is not None:
             jax_linear.bias.value = t2j(pt_linear.bias)
        else:
             print(f"Warning: PyTorch layer has bias but JAX layer does not. PT bias shape: {pt_linear.bias.shape}")

def check_kda_equivalence():
    # Configuration
    B, L = 2, 1024
    H = 64
    NH = 32
    NV = 32
    D = 128
    K = 4
    
    print(f"\n=== Testing Kimi Delta Attention Equivalence ===")
    print(f"Config: B={B}, L={L}, H={H}, NH={NH}, NV={NV}, D={D}, K={K}")
    
    # 1. Setup PyTorch (GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("PyTorch Device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("PyTorch Device: MPS")
    else:
        device = torch.device('cpu')
        print("PyTorch Device: CPU (Fallback)")

    torch.manual_seed(42)
    pt_model = TorchKDA(
        hidden_size=H,
        head_dim=D,
        num_heads=NH,
        num_v_heads=NV,
        mode='chunk',
        use_short_conv=True,
        conv_size=K,
        conv_bias=False,
        norm_eps=1e-5
    ).to(device).to(torch.bfloat16)
    pt_model.eval() 
    
    # 2. Setup JAX (CPU)
    os.environ['JAX_PLATFORMS'] = 'cpu'
    print("JAX Device: CPU")
    
    rngs = nnx.Rngs(0)
    jax_model = JaxKDA(
        hidden_size=H,
        num_heads=NH,
        head_dim=D,
        num_v_heads=NV,
        conv_kernel_size=K,
        normalization_layer_epsilon=1e-5,
        dtype=jnp.bfloat16,
        rngs=rngs
    )
    
    # 3. Transfer Weights (Torch -> JAX)
    print("Transferring weights...")
    
    # Projections
    set_linear(jax_model.q_proj, pt_model.q_proj)
    set_linear(jax_model.k_proj, pt_model.k_proj)
    set_linear(jax_model.v_proj, pt_model.v_proj)
    set_linear(jax_model.b_proj, pt_model.b_proj)
    set_linear(jax_model.o_proj, pt_model.o_proj)
    
    # Gate Projections (Sequential in Torch)
    set_linear(jax_model.f_a_proj, pt_model.f_proj[0])
    set_linear(jax_model.f_b_proj, pt_model.f_proj[1])
    set_linear(jax_model.g_a_proj, pt_model.g_proj[0])
    set_linear(jax_model.g_b_proj, pt_model.g_proj[1]) # Has bias
    
    # Convolutions
    # PT: (Out, In/G, K) -> (C, 1, K)
    # JAX: (K, In/G, Out) -> (K, 1, C)
    # Permute PT (2, 1, 0)
    jax_model.q_conv1d.kernel.value = t2j(pt_model.q_conv1d.weight.permute(2, 1, 0))
    jax_model.k_conv1d.kernel.value = t2j(pt_model.k_conv1d.weight.permute(2, 1, 0))
    jax_model.v_conv1d.kernel.value = t2j(pt_model.v_conv1d.weight.permute(2, 1, 0))
    
    # Parameters
    # A_log: PT (NH) -> JAX (1, 1, NH, 1)
    jax_model.A_log.value = t2j(pt_model.A_log).reshape(1, 1, NH, 1)
    jax_model.dt_bias.value = t2j(pt_model.dt_bias)
    
    # Norm
    jax_model.o_norm.rms_norm.scale.value = t2j(pt_model.o_norm.weight)
    
    # 4. Inputs
    np.random.seed(42)
    x_np = np.random.randn(B, L, H).astype(np.float32)
    dy_np = np.random.randn(B, L, H).astype(np.float32) # Random gradient
    
    x_pt = torch.tensor(x_np, device=device, dtype=torch.bfloat16, requires_grad=True)
    dy_pt = torch.tensor(dy_np, device=device, dtype=torch.bfloat16)
    
    x_jax = jnp.array(x_np, dtype=jnp.bfloat16)
    dy_jax = jnp.array(dy_np, dtype=jnp.bfloat16)
    
    # 5. Forward Check
    print("Running Forward Pass...")
    
    # PyTorch Forward
    # output_attentions=False returns (hidden_states, None, past_key_values)
    out_pt, _, _ = pt_model(x_pt, output_attentions=False)
    
    # JAX Forward
    out_jax, _ = jax_model(x_jax)
    
    # Compare
    out_pt_np = out_pt.detach().cpu().float().numpy()
    out_jax_np = np.array(out_jax, dtype=jnp.float32)

    if np.isnan(out_pt_np).any():
        print("❌ NaN detected in PyTorch Forward Output!")
        sys.exit(1)
    if np.isnan(out_jax_np).any():
        print("❌ NaN detected in JAX Forward Output!")
        sys.exit(1)
    
    diff = np.abs(out_pt_np - out_jax_np)
    print(f"Forward Output Shape: {out_pt_np.shape}")
    print(f"Forward Max Diff: {diff.max():.2e}")
    print(f"Forward Mean Diff: {diff.mean():.2e}")
    
    if diff.max() > 5e-2:
        print("❌ Forward pass mismatch!")
    else:
        print("✅ Forward pass matched!")

    # 6. Backward Check
    print("Running Backward Pass...")
    
    # PyTorch Backward
    # Loss = sum(out * dy)
    loss_pt = (out_pt * dy_pt).sum()
    loss_pt.backward()
    dx_pt = x_pt.grad
    
    # JAX Backward
    def loss_fn(model, x):
        out, _ = model(x)
        return jnp.sum(out * dy_jax)

    grad_fn = nnx.value_and_grad(loss_fn, argnums=1)
    loss_jax, dx_jax = grad_fn(jax_model, x_jax)
    
    # Compare Gradients
    dx_pt_np = dx_pt.cpu().float().numpy()
    dx_jax_np = np.array(dx_jax, dtype=jnp.float32)

    if np.isnan(dx_pt_np).any():
        print("❌ NaN detected in PyTorch Backward Gradient!")
        sys.exit(1)
    if np.isnan(dx_jax_np).any():
        print("❌ NaN detected in JAX Backward Gradient!")
        sys.exit(1)
    
    diff_grad = np.abs(dx_pt_np - dx_jax_np)
    print(f"Backward Grad Shape: {dx_pt_np.shape}")
    print(f"Backward Max Diff: {diff_grad.max():.2e}")
    print(f"Backward Mean Diff: {diff_grad.mean():.2e}")
    
    cosine_sim = np.sum(dx_pt_np * dx_jax_np) / (np.linalg.norm(dx_pt_np) * np.linalg.norm(dx_jax_np))
    print(f"Gradient Cosine Similarity: {cosine_sim:.8f}")
    
    if diff_grad.max() > 1e-1 and cosine_sim < 0.99:
        print("❌ Backward pass mismatch!")
    else:
        print("✅ Backward pass matched!")

if __name__ == "__main__":
    check_kda_equivalence()
