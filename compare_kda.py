
import os
import sys
import torch
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import math

# 1. Setup Environment and Paths
# -----------------------------------------------------------------------------
root_dir = os.getcwd()
fla_dir = os.path.join(root_dir, 'fla')
delta_dir = os.path.join(root_dir, 'delta_attention_comparison')

sys.path.append(fla_dir)
sys.path.append(delta_dir)

print("Imports setup...")
try:
    from fla.layers.kda import KimiDeltaAttention as TorchKDA
    # Fix for relative imports in JAX implementation:
    # We need 'src' to be importable as a top-level module
    from src.layers.kimi_delta_attention import KimiDeltaAttention as JaxKDA
except ImportError as e:
    print(f"Import failed: {e}")
    # Try to verify path
    print(f"sys.path: {sys.path}")
    sys.exit(1)

# 2. Configuration
# -----------------------------------------------------------------------------
BATCH_SIZE = 2
SEQ_LEN = 128
HIDDEN_SIZE = 256
NUM_HEADS = 4
HEAD_DIM = 64
CHUNK_SIZE = 64

# Ensure JAX uses GPU if available
print(f"JAX Default Backend: {jax.default_backend()}")
print(f"JAX Devices: {jax.devices()}")

# 3. Model Initialization
# -----------------------------------------------------------------------------
print("Initializing Models...")

# PyTorch Model
device = torch.device('cuda')
torch_model = TorchKDA(
    hidden_size=HIDDEN_SIZE,
    expand_v=1.0, # Default for safety (num_v_heads = num_heads)
    head_dim=HEAD_DIM,
    num_heads=NUM_HEADS,
    num_v_heads=NUM_HEADS,
    mode='chunk', # Training mode matches JAX chunk implementation
    use_short_conv=True,
    conv_size=4,
    conv_bias=False
).to(device)
torch_model.eval()

# JAX Model
rngs = nnx.Rngs(0)
jax_model = JaxKDA(
    hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS,
    head_dim=HEAD_DIM,
    num_v_heads=NUM_HEADS,
    conv_kernel_size=4,
    normalization_layer_epsilon=1e-5,
    dtype=jnp.float32,
    rngs=rngs,
)

# 4. Weight Transfer (PyTorch -> JAX)
# -----------------------------------------------------------------------------
print("Transferring Weights from PyTorch to JAX...")

def copy_linear(pt_linear, jax_dense):
    # PT: (out, in). JAX: (in, out)
    w = pt_linear.weight.detach().cpu().numpy()
    jax_dense.kernel.value = jnp.array(w.T)
    if pt_linear.bias is not None:
        b = pt_linear.bias.detach().cpu().numpy()
        jax_dense.bias.value = jnp.array(b)

def copy_conv(pt_conv, jax_conv):
    # PT: (D, 1, K). JAX: (K, 1, D) for depthwise
    w = pt_conv.weight.detach().cpu().numpy() # (D, 1, K)
    # Target: (K, 1, D) -> Permute (2, 1, 0)
    w_jax = np.transpose(w, (2, 1, 0))
    jax_conv.kernel.value = jnp.array(w_jax)
    if pt_conv.bias is not None:
        b = pt_conv.bias.detach().cpu().numpy()
        jax_conv.bias.value = jnp.array(b)

# Projections
copy_linear(torch_model.q_proj, jax_model.q_proj)
copy_linear(torch_model.k_proj, jax_model.k_proj)
copy_linear(torch_model.v_proj, jax_model.v_proj)
copy_linear(torch_model.o_proj, jax_model.o_proj)

# Convolutions
copy_conv(torch_model.q_conv1d, jax_model.q_conv1d)
copy_conv(torch_model.k_conv1d, jax_model.k_conv1d)
copy_conv(torch_model.v_conv1d, jax_model.v_conv1d)

# Gates Projections
# PT: f_proj (Sequential) -> JAX: f_proj (Sequential)
copy_linear(torch_model.f_proj[0], jax_model.f_proj.layers[0])
copy_linear(torch_model.f_proj[1], jax_model.f_proj.layers[1])

# PT: g_proj (Sequential) -> JAX: g_proj (Sequential)
copy_linear(torch_model.g_proj[0], jax_model.g_proj.layers[0])
copy_linear(torch_model.g_proj[1], jax_model.g_proj.layers[1])

# Beta Projection
copy_linear(torch_model.b_proj, jax_model.b_proj)

# Parameters
# A_log
# PT: (H). JAX: (1, 1, H, 1)
a_log_pt = torch_model.A_log.detach().cpu().numpy()
jax_model.A_log.value = jnp.array(a_log_pt.reshape(1, 1, NUM_HEADS, 1))

# dt_bias
# PT: (D). JAX: (D)
dt_bias_pt = torch_model.dt_bias.detach().cpu().numpy()
jax_model.dt_bias.value = jnp.array(dt_bias_pt)

    
# Output Norm (FusedRMSNormGated)
# PT: o_norm.weight -> JAX: o_norm.rms_norm.scale
# Note: PT's FusedRMSNormGated usually inherits from RMSNorm or implements it similarly.
# Let's check if it has 'weight' or 'scale'. Usually 'weight'.
norm_weight = torch_model.o_norm.weight.detach().cpu().numpy()
jax_model.o_norm.rms_norm.scale.value = jnp.array(norm_weight)


# 5. Run Comparison
# -----------------------------------------------------------------------------
print("Running Forward Pass...")

# Generate Input
np.random.seed(42)
x_np = np.random.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).astype(np.float32)

# PyTorch Run
x_pt = torch.tensor(x_np, device=device)
with torch.no_grad():
    y_pt, _, _ = torch_model(x_pt)
    y_pt_np = y_pt.cpu().numpy()

# JAX Run
x_jax = jnp.array(x_np)
# JAX Call: (hidden_states, chunk_size, ...)
# Note: output is (output, final_state, past_key_values)
y_jax, _, _ = jax_model(x_jax, chunk_size=CHUNK_SIZE)
y_jax_np = np.array(y_jax)

# 6. Analysis
# -----------------------------------------------------------------------------
print(f"PyTorch Output Shape: {y_pt_np.shape}")
print(f"JAX Output Shape:     {y_jax_np.shape}")

diff = np.abs(y_pt_np - y_jax_np)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print("-" * 40)
print(f"Max Absolute Difference: {max_diff:.6f}")
print(f"Mean Absolute Difference: {mean_diff:.6f}")
print("-" * 40)

if max_diff < 1e-3:
    print("SUCCESS: Results match closely!")
else:
    print("WARNING: Results diverge!")
    # Debug info
    print("PyTorch Sample (0,0,:5):", y_pt_np[0,0,:5])
    print("JAX Sample (0,0,:5):    ", y_jax_np[0,0,:5])

# Additional: Initial States or Masks were not used in this simple test.
# Both models used default initial states (None -> zeros) and full attention (no mask).
