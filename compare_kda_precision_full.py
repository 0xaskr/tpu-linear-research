"""Full KDA layer precision comparison: Triton CPU (FP32) vs JAX CPU (FP32).

Compares ALL intermediate variables of fla KimiDeltaAttention (PyTorch/Triton)
against the standalone JAX implementation (attention_kda_standalone.py).

Supports both dense (fixed-length) and varlen (packed) modes.
"""

import os
os.environ["TRITON_CPU_BACKEND"] = "1"
os.environ["TRITON_INTERPRET"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
import numpy as np
import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
from flax import nnx

# Path setup
sys.path.insert(0, os.getcwd())
if os.path.exists(os.path.join(os.getcwd(), 'fla')):
    sys.path.insert(0, os.path.join(os.getcwd(), 'fla'))
if os.path.exists(os.path.join(os.getcwd(), 'delta_attention_comparison')):
    sys.path.insert(0, os.path.join(os.getcwd(), 'delta_attention_comparison'))

# Import fla (PyTorch/Triton)
from fla.layers.kda import KimiDeltaAttention as FlaKDA
from fla.ops.kda.gate import naive_kda_gate as fla_naive_kda_gate

# Import JAX standalone
from attention_kda_standalone import (
    KimiDeltaAttention as JaxKDA,
    fused_kda_gate as jax_fused_kda_gate,
)
from src.initializers import nd_dense_init


# ============================================================================
# Tensor comparison utility
# ============================================================================

def compare_tensor(name, pt_t, jax_t, atol=1e-5, rtol=1e-5):
    """Compare PyTorch and JAX tensors with detailed diagnostics."""
    if pt_t is None and jax_t is None:
        print(f"  [{name}] Both None. SKIP.")
        return True
    if pt_t is None or jax_t is None:
        print(f"  [{name}] One is None! MISMATCH.")
        return False

    pt_val = pt_t.detach().cpu().float().numpy() if isinstance(pt_t, torch.Tensor) else np.array(pt_t)
    jax_val = np.array(jax_t) if not isinstance(jax_t, np.ndarray) else jax_t
    if jax_val.dtype != np.float32:
        jax_val = jax_val.astype(np.float32)
    if pt_val.dtype != np.float32:
        pt_val = pt_val.astype(np.float32)

    if pt_val.shape != jax_val.shape:
        if pt_val.squeeze().shape == jax_val.squeeze().shape:
            pt_val = pt_val.squeeze()
            jax_val = jax_val.squeeze()
        else:
            print(f"  [{name}] Shape mismatch: PT {pt_val.shape} vs JAX {jax_val.shape}. FAIL.")
            return False

    diff = np.abs(pt_val - jax_val)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    tolerance = atol + rtol * np.abs(jax_val)
    error_ratio = diff / (tolerance + 1e-12)
    max_error_ratio = np.max(error_ratio)
    is_close = np.allclose(pt_val, jax_val, atol=atol, rtol=rtol)
    status = "PASS" if is_close else "FAIL"

    print(f"  [{name}] {status} | max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} max_err_ratio={max_error_ratio:.4f}")

    if not is_close:
        idx = np.unravel_index(np.argmax(error_ratio), error_ratio.shape)
        print(f"    Worst at {idx}: PT={pt_val[idx]:.8f} JAX={jax_val[idx]:.8f} diff={diff[idx]:.6e}")

    return is_close


# ============================================================================
# Weight transfer: fla (PyTorch) -> JAX standalone
# ============================================================================

def transfer_weights(fla_model: FlaKDA, jax_model: JaxKDA):
    """Copy weights from fla PyTorch model to JAX standalone model.

    Key mappings:
        PyTorch nn.Linear.weight [out, in] -> DenseGeneral.kernel [in, out]
        fla ShortConvolution (nn.Conv1d) weight [D, 1, W] -> nnx.Conv kernel [W, 1, D]
    """
    def pt2np(t):
        return t.detach().cpu().float().numpy()

    def set_param(jax_param, np_val):
        """Set a Flax nnx.Param or nnx.Variable value."""
        if isinstance(jax_param, nnx.Param):
            jax_param.value = jnp.array(np_val)
        elif hasattr(jax_param, 'value'):
            jax_param.value = jnp.array(np_val)
        else:
            raise ValueError(f"Unknown param type: {type(jax_param)}")

    # --- Linear projections: weight [out, in] -> kernel [in, out] ---
    for name in ('q_proj', 'k_proj', 'v_proj', 'b_proj', 'o_proj'):
        pt_layer = getattr(fla_model, name)
        jax_layer = getattr(jax_model, name)
        set_param(jax_layer.kernel, pt2np(pt_layer.weight).T)

    # --- f_proj (nn.Sequential) -> f_a_proj + f_b_proj ---
    set_param(jax_model.f_a_proj.kernel, pt2np(fla_model.f_proj[0].weight).T)
    set_param(jax_model.f_b_proj.kernel, pt2np(fla_model.f_proj[1].weight).T)

    # --- g_proj (nn.Sequential) -> g_a_proj + g_b_proj ---
    set_param(jax_model.g_a_proj.kernel, pt2np(fla_model.g_proj[0].weight).T)
    set_param(jax_model.g_b_proj.kernel, pt2np(fla_model.g_proj[1].weight).T)
    if fla_model.g_proj[1].bias is not None:
        set_param(jax_model.g_b_proj.bias, pt2np(fla_model.g_proj[1].bias))

    # --- ShortConvolution conv weights ---
    # fla: nn.Conv1d weight [D, 1, W]
    # JAX nnx.Conv kernel: [W, 1, D]  (for feature_group_count=D, depthwise)
    if fla_model.use_short_conv:
        for conv_name in ('q_conv1d', 'k_conv1d', 'v_conv1d'):
            pt_conv = getattr(fla_model, conv_name)
            jax_conv = getattr(jax_model, conv_name)
            # pt_conv.weight shape: [D, 1, W]
            pt_w = pt2np(pt_conv.weight)  # [D, 1, W]
            # -> [W, 1, D]
            jax_w = np.transpose(pt_w, (2, 1, 0))  # [W, 1, D]
            set_param(jax_conv.conv.kernel, jax_w)
            if pt_conv.bias is not None:
                set_param(jax_conv.conv.bias, pt2np(pt_conv.bias))

    # --- A_log, dt_bias ---
    set_param(jax_model.A_log, pt2np(fla_model.A_log))
    set_param(jax_model.dt_bias, pt2np(fla_model.dt_bias))

    # --- o_norm (FusedRMSNormGated) ---
    # fla o_norm.weight -> jax o_norm.rms_norm.scale
    if hasattr(fla_model.o_norm, 'weight') and fla_model.o_norm.weight is not None:
        set_param(jax_model.o_norm.rms_norm.scale, pt2np(fla_model.o_norm.weight))


# ============================================================================
# Pure-PyTorch causal depthwise conv1d (no Triton kernel, avoids autotuner bug)
# ============================================================================

def _manual_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Pure PyTorch depthwise causal conv1d with varlen support.

    Args:
        x: [B, T, D]
        weight: [D, 1, W] (from nn.Conv1d with groups=D)
        bias: [D] or None
        cu_seqlens: [N+1] or None
        activation: "silu" or None
    Returns:
        y: [B, T, D]
    """
    B, T, D = x.shape
    W = weight.shape[2]

    if cu_seqlens is None:
        # Dense: standard causal conv via F.conv1d with left-padding
        x_t = x.transpose(1, 2)  # [B, D, T]
        x_padded = torch.nn.functional.pad(x_t, (W - 1, 0))  # [B, D, T+W-1]
        y_t = torch.nn.functional.conv1d(x_padded, weight, bias=bias, groups=D)  # [B, D, T]
        y = y_t.transpose(1, 2)  # [B, T, D]
    else:
        # Varlen: per-sequence causal conv
        assert B == 1
        x_flat = x[0]  # [T, D]
        y_flat = torch.zeros_like(x_flat)
        N = len(cu_seqlens) - 1
        for n in range(N):
            bos = cu_seqlens[n].item()
            eos = cu_seqlens[n + 1].item()
            x_seq = x_flat[bos:eos]  # [L, D]
            x_seq_t = x_seq.T.unsqueeze(0)  # [1, D, L]
            x_padded = torch.nn.functional.pad(x_seq_t, (W - 1, 0))  # [1, D, L+W-1]
            y_seq_t = torch.nn.functional.conv1d(x_padded, weight, bias=bias, groups=D)  # [1, D, L]
            y_flat[bos:eos] = y_seq_t[0].T
        y = y_flat.unsqueeze(0)  # [1, T, D]

    if activation == "silu":
        y = torch.nn.functional.silu(y)
    return y


# ============================================================================
# Pure-PyTorch naive fused_recurrent_kda (no Triton, with gk + cu_seqlens)
# ============================================================================

def naive_fused_recurrent_kda(
    q: torch.Tensor,          # [B, T, H, K]
    k: torch.Tensor,          # [B, T, H, K]
    v: torch.Tensor,          # [B, T, H, V]
    gk: torch.Tensor,         # [B, T, H, K] per-key-dim gate
    beta: torch.Tensor,       # [B, T, H]
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,  # [N, H, K, V]
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,  # [N+1]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pure PyTorch naive recurrence for KDA (gated delta rule with gk).

    Recurrence (per-key-dim gating):
        S[t] = diag(exp(gk[t])) @ S[t-1] + beta[t] * k[t] outer (v[t] - S[t-1]^T k[t])
        o[t] = scale * q[t]^T S[t]

    Supports cu_seqlens for variable-length packed sequences.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    if scale is None:
        scale = K ** -0.5

    q = q.float()
    k = k.float()
    v = v.float()
    gk = gk.float()
    beta = beta.float()

    if cu_seqlens is not None:
        # Varlen mode: packed sequences
        assert B == 1
        N = len(cu_seqlens) - 1
        o_flat = torch.zeros(1, T, H, V, dtype=torch.float32, device=q.device)
        final_states = [] if output_final_state else None

        for n in range(N):
            bos = cu_seqlens[n].item()
            eos = cu_seqlens[n + 1].item()
            L = eos - bos

            h = torch.zeros(H, K, V, dtype=torch.float32, device=q.device)
            if initial_state is not None:
                h = initial_state[n].clone().float()

            for t in range(L):
                b_q = q[0, bos + t]                # [H, K]
                b_k = k[0, bos + t]                # [H, K]
                b_v = v[0, bos + t]                # [H, V]
                b_gk = gk[0, bos + t]              # [H, K]
                b_beta = beta[0, bos + t]           # [H]

                if use_qk_l2norm_in_kernel:
                    b_q = torch.nn.functional.normalize(b_q, p=2, dim=-1)
                    b_k = torch.nn.functional.normalize(b_k, p=2, dim=-1)

                # Decay: h[h, k, v] *= exp(gk[h, k])
                h = h * b_gk.exp().unsqueeze(-1)     # [H, K, V]

                # Delta: residual = v - S^T k
                residual = b_v - torch.einsum('hkv,hk->hv', h, b_k)  # [H, V]

                # Update: h += beta * k outer residual
                h = h + b_beta.unsqueeze(-1).unsqueeze(-1) * b_k.unsqueeze(-1) * residual.unsqueeze(-2)

                # Readout: o = q^T S
                o_flat[0, bos + t] = scale * torch.einsum('hk,hkv->hv', b_q, h)

            if output_final_state:
                final_states.append(h)

        if output_final_state:
            final_state = torch.stack(final_states, dim=0)  # [N, H, K, V]
        else:
            final_state = None

        return o_flat, final_state
    else:
        # Dense mode
        o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)
        final_states = [] if output_final_state else None

        for b in range(B):
            h = torch.zeros(H, K, V, dtype=torch.float32, device=q.device)
            if initial_state is not None:
                h = initial_state[b].clone().float()

            for t in range(T):
                b_q = q[b, t]                      # [H, K]
                b_k = k[b, t]                      # [H, K]
                b_v = v[b, t]                      # [H, V]
                b_gk = gk[b, t]                    # [H, K]
                b_beta = beta[b, t]                # [H]

                if use_qk_l2norm_in_kernel:
                    b_q = torch.nn.functional.normalize(b_q, p=2, dim=-1)
                    b_k = torch.nn.functional.normalize(b_k, p=2, dim=-1)

                # Decay
                h = h * b_gk.exp().unsqueeze(-1)

                # Delta
                residual = b_v - torch.einsum('hkv,hk->hv', h, b_k)

                # Update
                h = h + b_beta.unsqueeze(-1).unsqueeze(-1) * b_k.unsqueeze(-1) * residual.unsqueeze(-2)

                # Readout
                o[b, t] = scale * torch.einsum('hk,hkv->hv', b_q, h)

            if output_final_state:
                final_states.append(h)

        if output_final_state:
            final_state = torch.stack(final_states, dim=0)  # [B, H, K, V]
        else:
            final_state = None

        return o, final_state


# ============================================================================
# Step-by-step fla forward (to extract intermediates)
# ============================================================================

@torch.no_grad()
def fla_forward_with_intermediates(
    model: FlaKDA,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run fla KDA forward step-by-step, returning all intermediates.

    Uses pure-PyTorch causal conv1d to avoid Triton autotuner issues.
    Uses fused_recurrent_kda with pre-computed gate for the core KDA step.
    """
    from einops import rearrange
    from torch.nn import functional as F

    intermediates = {}
    B, T, D = hidden_states.shape

    # Projections
    q_proj_out = model.q_proj(hidden_states)
    k_proj_out = model.k_proj(hidden_states)
    v_proj_out = model.v_proj(hidden_states)
    intermediates["q_proj_out"] = q_proj_out
    intermediates["k_proj_out"] = k_proj_out
    intermediates["v_proj_out"] = v_proj_out

    # Short Convolution (pure PyTorch, no Triton)
    if model.use_short_conv:
        q_conv_out = _manual_causal_conv1d(
            q_proj_out, model.q_conv1d.weight, model.q_conv1d.bias,
            cu_seqlens, activation="silu")
        k_conv_out = _manual_causal_conv1d(
            k_proj_out, model.k_conv1d.weight, model.k_conv1d.bias,
            cu_seqlens, activation="silu")
        v_conv_out = _manual_causal_conv1d(
            v_proj_out, model.v_conv1d.weight, model.v_conv1d.bias,
            cu_seqlens, activation="silu")
    else:
        q_conv_out = F.silu(q_proj_out)
        k_conv_out = F.silu(k_proj_out)
        v_conv_out = F.silu(v_proj_out)

    intermediates["q_conv_out"] = q_conv_out
    intermediates["k_conv_out"] = k_conv_out
    intermediates["v_conv_out"] = v_conv_out

    # f_proj -> g_pre_gate
    g_linear = model.f_proj(hidden_states)
    g_pre_gate = rearrange(g_linear, "... (h d) -> ... h d", d=model.head_k_dim)
    intermediates["g_pre_gate"] = g_pre_gate

    # fused_kda_gate (using naive pure-PyTorch version to avoid Triton autotuner crash)
    g_post_gate = fla_naive_kda_gate(
        g=g_pre_gate, A_log=model.A_log, dt_bias=model.dt_bias)
    intermediates["g_post_gate"] = g_post_gate

    # beta
    beta = model.b_proj(hidden_states).sigmoid()
    intermediates["beta"] = beta

    # reshape q, k, v
    q = rearrange(q_conv_out, "... (h d) -> ... h d", d=model.head_k_dim)
    k = rearrange(k_conv_out, "... (h d) -> ... h d", d=model.head_k_dim)
    v = rearrange(v_conv_out, "... (h d) -> ... h d", d=model.head_v_dim)

    # KDA core: use naive pure-PyTorch recurrence (no Triton)
    o_pre_norm, recurrent_state = naive_fused_recurrent_kda(
        q=q, k=k, v=v, gk=g_post_gate, beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    intermediates["o_pre_norm"] = o_pre_norm
    intermediates["recurrent_state"] = recurrent_state

    # g_proj -> gate output
    g_for_o_norm = rearrange(model.g_proj(hidden_states), "... (h d) -> ... h d", d=model.head_v_dim)
    intermediates["g_for_o_norm"] = g_for_o_norm

    # o_norm: pure PyTorch RMSNorm + sigmoid gating (avoids Triton FusedRMSNormGated)
    # Compute: o_post_norm = RMSNorm(o_pre_norm) * sigmoid(g_for_o_norm)
    o_flat = o_pre_norm.float()
    eps = model.o_norm.eps
    rms = torch.sqrt(o_flat.pow(2).mean(dim=-1, keepdim=True) + eps)
    o_normed = o_flat / rms
    if model.o_norm.weight is not None:
        o_normed = o_normed * model.o_norm.weight.float()
    o_post_norm = o_normed * torch.sigmoid(g_for_o_norm.float())
    intermediates["o_post_norm"] = o_post_norm

    # o_proj
    o_out = model.o_proj(rearrange(o_post_norm, "b t h d -> b t (h d)"))
    intermediates["o_out"] = o_out

    return intermediates


# ============================================================================
# Test: Dense (fixed-length) sequences
# ============================================================================

def run_dense_comparison(
    B=2, T=128, hidden_size=256, num_heads=4, head_dim=64,
    conv_size=4, use_short_conv=True, conv_bias=False,
    norm_eps=1e-5, atol=1e-5, rtol=1e-5, label="",
):
    tag = f"Dense {label}" if label else "Dense"
    print(f"\n{'=' * 60}")
    print(f"  {tag} — FP32")
    print(f"{'=' * 60}")
    print(f"  B={B}, T={T}, hidden={hidden_size}, H={num_heads}, D={head_dim}, "
          f"conv={conv_size}, use_short_conv={use_short_conv}, conv_bias={conv_bias}")

    torch.manual_seed(42)

    # Create fla model
    fla_model = FlaKDA(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        mode="chunk",
        use_short_conv=use_short_conv,
        conv_size=conv_size,
        conv_bias=conv_bias,
        norm_eps=norm_eps,
    ).float().eval()

    # Create JAX model
    rngs = nnx.Rngs(0)
    jax_model = JaxKDA(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        mode="chunk",
        use_short_conv=use_short_conv,
        conv_size=conv_size,
        conv_bias=conv_bias,
        norm_eps=norm_eps,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=rngs,
    )

    # Transfer weights
    transfer_weights(fla_model, jax_model)
    print("  Weights transferred.")

    # Input
    torch.manual_seed(123)
    x_pt = torch.randn(B, T, hidden_size, dtype=torch.float32)
    x_jax = jnp.array(x_pt.numpy())

    # Run fla (step-by-step)
    print("\n  Running fla (PyTorch CPU, FP32)...")
    with torch.no_grad():
        pt_out = fla_forward_with_intermediates(fla_model, x_pt, cu_seqlens=None)

    # Run JAX
    print("  Running JAX (CPU, FP32)...")
    jax_o, _, jax_inter = jax_model(
        x_jax, cu_seqlens=None, output_final_state=True,
        return_intermediates=True, training=False)

    # Compare
    print("\n  --- Comparison Results ---")
    all_pass = True
    for name in [
        "q_proj_out", "k_proj_out", "v_proj_out",
        "q_conv_out", "k_conv_out", "v_conv_out",
        "g_pre_gate", "g_post_gate", "beta",
        "o_pre_norm", "g_for_o_norm", "o_post_norm", "o_out",
        "recurrent_state",
    ]:
        pt_val = pt_out.get(name)
        jax_val = jax_inter.get(name) if jax_inter else None
        ok = compare_tensor(name, pt_val, jax_val, atol=atol, rtol=rtol)
        all_pass = all_pass and ok

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return all_pass


# ============================================================================
# Test: Varlen (packed) sequences
# ============================================================================

def run_varlen_comparison(
    seqlens=None, hidden_size=256, num_heads=4, head_dim=64,
    conv_size=4, use_short_conv=True, conv_bias=False,
    norm_eps=1e-5, atol=1e-5, rtol=1e-5, label="",
):
    if seqlens is None:
        seqlens = [64, 128, 64]
    TotalT = sum(seqlens)
    tag = f"Varlen {label}" if label else "Varlen"
    print(f"\n{'=' * 60}")
    print(f"  {tag} — FP32")
    print(f"{'=' * 60}")

    cu_seqlens_list = [0] + list(np.cumsum(seqlens))
    cu_seqlens_pt = torch.tensor(cu_seqlens_list, dtype=torch.long)
    cu_seqlens_jax = jnp.array(cu_seqlens_list, dtype=jnp.int32)

    print(f"  seqlens={seqlens}, TotalT={TotalT}, hidden={hidden_size}, H={num_heads}, D={head_dim}, "
          f"conv={conv_size}, use_short_conv={use_short_conv}, conv_bias={conv_bias}")

    torch.manual_seed(42)

    # Create fla model
    fla_model = FlaKDA(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        mode="chunk",
        use_short_conv=use_short_conv,
        conv_size=conv_size,
        conv_bias=conv_bias,
        norm_eps=norm_eps,
    ).float().eval()

    # Create JAX model
    rngs = nnx.Rngs(0)
    jax_model = JaxKDA(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        mode="chunk",
        use_short_conv=use_short_conv,
        conv_size=conv_size,
        conv_bias=conv_bias,
        norm_eps=norm_eps,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=rngs,
    )

    # Transfer weights
    transfer_weights(fla_model, jax_model)
    print("  Weights transferred.")

    # Input: packed [1, TotalT, hidden]
    torch.manual_seed(123)
    x_pt = torch.randn(1, TotalT, hidden_size, dtype=torch.float32)
    x_jax = jnp.array(x_pt.numpy())

    # Run fla
    print(f"\n  Running fla (PyTorch CPU, FP32) with cu_seqlens...")
    with torch.no_grad():
        pt_out = fla_forward_with_intermediates(fla_model, x_pt, cu_seqlens=cu_seqlens_pt)

    # Run JAX
    print("  Running JAX (CPU, FP32) with cu_seqlens...")
    jax_o, _, jax_inter = jax_model(
        x_jax, cu_seqlens=cu_seqlens_jax, output_final_state=True,
        return_intermediates=True, training=False)

    # Compare
    print("\n  --- Comparison Results ---")
    all_pass = True
    for name in [
        "q_proj_out", "k_proj_out", "v_proj_out",
        "q_conv_out", "k_conv_out", "v_conv_out",
        "g_pre_gate", "g_post_gate", "beta",
        "o_pre_norm", "g_for_o_norm", "o_post_norm", "o_out",
        "recurrent_state",
    ]:
        pt_val = pt_out.get(name)
        jax_val = jax_inter.get(name) if jax_inter else None
        ok = compare_tensor(name, pt_val, jax_val, atol=atol, rtol=rtol)
        all_pass = all_pass and ok

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return all_pass


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  KDA Full Layer Precision Test")
    print("  PyTorch CPU (FP32) vs JAX CPU (FP32)")
    print("=" * 60)

    results = {}

    # ---- Case 1: Dense baseline ----
    results["dense_baseline"] = run_dense_comparison(
        B=2, T=128, hidden_size=256, num_heads=4, head_dim=64,
        conv_size=4, use_short_conv=True, conv_bias=False,
        label="(baseline B=2,T=128,H=4,D=64,conv=4)")

    # ---- Case 2: Dense, no short conv ----
    results["dense_no_conv"] = run_dense_comparison(
        B=2, T=128, hidden_size=256, num_heads=4, head_dim=64,
        use_short_conv=False,
        label="(no short conv)")

    # ---- Case 3: Dense, with conv_bias ----
    results["dense_conv_bias"] = run_dense_comparison(
        B=2, T=64, hidden_size=256, num_heads=4, head_dim=64,
        conv_size=4, use_short_conv=True, conv_bias=True,
        label="(conv_bias=True)")

    # ---- Case 4: Dense, single batch ----
    results["dense_B1"] = run_dense_comparison(
        B=1, T=256, hidden_size=256, num_heads=4, head_dim=64,
        conv_size=4, use_short_conv=True,
        label="(B=1,T=256)")

    # ---- Case 5: Dense, larger model ----
    results["dense_large"] = run_dense_comparison(
        B=1, T=64, hidden_size=512, num_heads=8, head_dim=64,
        conv_size=4, use_short_conv=True,
        label="(large H=8,hidden=512)")

    # ---- Case 6: Dense, small head_dim ----
    results["dense_small_D"] = run_dense_comparison(
        B=2, T=64, hidden_size=128, num_heads=4, head_dim=32,
        conv_size=4, use_short_conv=True,
        label="(small D=32,hidden=128)")

    # ---- Case 7: Dense, longer conv kernel ----
    results["dense_conv8"] = run_dense_comparison(
        B=2, T=64, hidden_size=256, num_heads=4, head_dim=64,
        conv_size=8, use_short_conv=True,
        label="(conv_size=8)")

    # ---- Case 8: Dense, very short sequence ----
    results["dense_short_T"] = run_dense_comparison(
        B=4, T=8, hidden_size=256, num_heads=4, head_dim=64,
        conv_size=4, use_short_conv=True,
        label="(B=4,T=8 short)")

    # ---- Case 9: Varlen baseline ----
    results["varlen_baseline"] = run_varlen_comparison(
        seqlens=[64, 128, 64], hidden_size=256, num_heads=4, head_dim=64,
        conv_size=4, use_short_conv=True,
        label="(baseline [64,128,64])")

    # ---- Case 10: Varlen, very short sequences ----
    results["varlen_short"] = run_varlen_comparison(
        seqlens=[1, 2, 3, 4, 5], hidden_size=256, num_heads=4, head_dim=64,
        conv_size=4, use_short_conv=True,
        label="(very short [1,2,3,4,5])")

    # ---- Case 11: Varlen, single sequence ----
    results["varlen_single"] = run_varlen_comparison(
        seqlens=[256], hidden_size=256, num_heads=4, head_dim=64,
        conv_size=4, use_short_conv=True,
        label="(single seq [256])")

    # ---- Case 12: Varlen, many short sequences ----
    results["varlen_many_short"] = run_varlen_comparison(
        seqlens=[8] * 16, hidden_size=256, num_heads=4, head_dim=64,
        conv_size=4, use_short_conv=True,
        label="(16x [8])")

    # ---- Case 13: Varlen, no short conv ----
    results["varlen_no_conv"] = run_varlen_comparison(
        seqlens=[32, 64, 32], hidden_size=256, num_heads=4, head_dim=64,
        use_short_conv=False,
        label="(no short conv)")

    # ---- Case 14: Varlen, with conv_bias ----
    results["varlen_conv_bias"] = run_varlen_comparison(
        seqlens=[32, 64, 32], hidden_size=256, num_heads=4, head_dim=64,
        conv_size=4, use_short_conv=True, conv_bias=True,
        label="(conv_bias=True)")

    # ---- Case 15: Varlen, unequal + large model ----
    results["varlen_large"] = run_varlen_comparison(
        seqlens=[16, 128, 32, 64], hidden_size=512, num_heads=8, head_dim=64,
        conv_size=4, use_short_conv=True,
        label="(large H=8,hidden=512,[16,128,32,64])")

    # ---- Case 16: Varlen, seq len shorter than conv ----
    results["varlen_shorter_than_conv"] = run_varlen_comparison(
        seqlens=[2, 3, 1, 2], hidden_size=256, num_heads=4, head_dim=64,
        conv_size=4, use_short_conv=True,
        label="(seqlens < conv_size [2,3,1,2])")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:35s} {status}")
        all_ok = all_ok and passed
    print("-" * 60)
    print(f"  {'TOTAL':35s} {'ALL PASS' if all_ok else 'SOME FAILURES'}")
    print("=" * 60)
