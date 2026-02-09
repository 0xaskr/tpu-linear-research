# Copyright 2023-2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone JAX KimiDelta Attention (KDA) with varlen short conv support.

No MaxText dependencies. Uses standalone DenseGeneral/l2norm/RMSNorm from
delta_attention_comparison/src/.  Aligned with fla/fla/layers/kda.py (PyTorch).
"""

from __future__ import annotations

import math
import sys
import os
from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

# ---------------------------------------------------------------------------
# Standalone imports from delta_attention_comparison/src/
# ---------------------------------------------------------------------------
_src_dir = os.path.join(os.path.dirname(__file__), "delta_attention_comparison")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from src.linears import DenseGeneral  # noqa: E402
from src.normalizations import l2norm, RMSNorm  # noqa: E402
from src.initializers import nd_dense_init, NdInitializer  # noqa: E402

# ---------------------------------------------------------------------------
# Type aliases (replaces MaxText.common_types)
# ---------------------------------------------------------------------------
Array = jax.Array
DType = Any
Config = Any


# ============================================================================
# Utility functions
# ============================================================================

def _softplus_stable(x: Array) -> Array:
    x_clip = jnp.clip(x, a_min=-20.0, a_max=20.0)
    return jnp.log1p(jnp.exp(x_clip))


def fused_kda_gate(
    g: Array,
    a_log: Array,
    dt_bias: Array | None = None,
    lower_bound: float | None = None,
    output_dtype: jnp.dtype = jnp.float32,
) -> Array:
    """Reference fused_kda_gate.

    Computes: g_post = -exp(a_log) * softplus(g + dt_bias)
    where a_log is broadcast to [1,1,H,1] and dt_bias to [1,1,H,D].
    """
    g_fp32 = g.astype(jnp.float32)
    num_heads = g_fp32.shape[-2]
    dim = g_fp32.shape[-1]

    a_log_values = a_log.reshape((num_heads,))
    a_exp = jnp.exp(a_log_values).reshape((1, 1, num_heads, 1))

    if dt_bias is not None:
        dt_bias = dt_bias.reshape((num_heads, dim))
        g_fp32 = g_fp32 + dt_bias.reshape((1, 1, num_heads, dim))

    if lower_bound is None:
        g_post = -a_exp * _softplus_stable(g_fp32)
    else:
        g_post = lower_bound * jax.nn.sigmoid(a_exp * g_fp32)

    return g_post.astype(output_dtype)


def fused_rms_norm_gated(
    x: Array,
    g: Array,
    weight: Array | None,
    eps: float,
    activation: str = "sigmoid",
) -> Array:
    """RMSNorm with gating, matching fla LayerNormGatedFunction (RMS path)."""
    x_fp32 = x.astype(jnp.float32)
    g_fp32 = g.astype(jnp.float32)

    d = x_fp32.shape[-1]
    x_2d = x_fp32.reshape(-1, d)
    g_2d = g_fp32.reshape(-1, d)

    rstd = jax.lax.rsqrt(jnp.mean(x_2d * x_2d, axis=-1, keepdims=True) + eps)
    y = x_2d * rstd
    if weight is not None:
        y = y * weight.astype(jnp.float32)

    if activation in ("swish", "silu"):
        gate = g_2d * jax.nn.sigmoid(g_2d)
    elif activation == "sigmoid":
        gate = jax.nn.sigmoid(g_2d)
    else:
        gate = g_2d

    y = y * gate
    y = y.reshape(x_fp32.shape)
    return y.astype(x.dtype)


# ============================================================================
# Varlen Short Convolution
# ============================================================================

class ShortConvolution(nnx.Module):
    """Depthwise causal 1D convolution with cu_seqlens (varlen) support.

    For varlen:
      - Input x is packed [1, TotalT, D] with cu_seqlens [N+1]
      - Each sequence is convolved independently (no cross-sequence leakage)
      - Uses masking approach for JAX/XLA compatibility (static shapes)
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        use_bias: bool = False,
        dtype: DType = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nnx.Conv(
            in_features=hidden_size,
            out_features=hidden_size,
            kernel_size=(kernel_size,),
            feature_group_count=hidden_size,
            padding="CAUSAL",
            use_bias=use_bias,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        x: Array,
        cache: Array | None = None,
        output_final_state: bool = False,
        cu_seqlens: Array | None = None,
    ) -> tuple[Array, Array | None]:
        """Forward pass.

        Args:
            x: [B, T, D] input. When cu_seqlens is provided, B must be 1 and
               T = total tokens across all packed sequences.
            cache: previous conv state [N, kernel_size-1, D] or None.
            output_final_state: whether to return final conv states.
            cu_seqlens: [N+1] cumulative sequence lengths (int32).

        Returns:
            y: [B, T, D] output (after SiLU activation).
            new_cache: [N, kernel_size-1, D] or None.
        """
        if cu_seqlens is not None:
            return self._forward_varlen(x, cache, output_final_state, cu_seqlens)
        else:
            return self._forward_dense(x, cache, output_final_state)

    def _forward_dense(
        self, x: Array, cache: Array | None, output_final_state: bool
    ) -> tuple[Array, Array | None]:
        """Non-varlen path: standard causal conv."""
        if cache is not None:
            x_full = jnp.concatenate([cache, x], axis=1)
            y_full = self.conv(x_full)
            y = y_full[:, -x.shape[1]:, :]
        else:
            x_full = x
            y = self.conv(x)

        y = jax.nn.silu(y.astype(jnp.float32)).astype(x.dtype)

        new_cache = None
        if output_final_state and self.kernel_size > 1:
            new_cache = x_full[:, -(self.kernel_size - 1):, :]
        return y, new_cache

    def _forward_varlen(
        self,
        x: Array,
        cache: Array | None,
        output_final_state: bool,
        cu_seqlens: Array,
    ) -> tuple[Array, Array | None]:
        """Varlen path: masking-based approach.

        Builds a window of neighbors for each token and masks out positions
        that cross sequence boundaries, then does depthwise conv + SiLU.
        """
        W = self.kernel_size
        B, T, D = x.shape
        assert B == 1, "Varlen requires batch size 1"
        x_flat = x[0]  # [T, D]

        # Build sequence-start indicators and seq_ids
        N = cu_seqlens.shape[0] - 1
        starts = jnp.zeros((T,), dtype=jnp.bool_)
        starts = starts.at[cu_seqlens[:-1]].set(True)
        seq_ids = jnp.cumsum(starts.astype(jnp.int32)) - 1  # [T], values in [0, N-1]

        # For each token t, the start of its sequence
        seq_starts = cu_seqlens[seq_ids]  # [T]

        # Extract conv kernel weights: shape [W, 1, D] for depthwise conv
        # We want weight[w, d] for each kernel tap w and channel d
        conv_kernel = self.conv.kernel.value  # [W, 1, D]  (feature_group_count=D)
        # Squeeze to [W, D]
        w_kernel = conv_kernel.reshape(W, D)

        # Build neighbor indices for causal conv:
        # For position t, we need x[t - (W-1) + w] for w = 0..W-1
        # i.e., positions t - (W-1), t - (W-2), ..., t
        offsets = jnp.arange(W) - (W - 1)  # [-(W-1), ..., 0]
        positions = jnp.arange(T)[:, None] + offsets[None, :]  # [T, W]

        # Mask: position is valid if >= seq_start[t] AND >= 0
        valid = (positions >= seq_starts[:, None]) & (positions >= 0)  # [T, W]

        # Clamp positions to [0, T-1] (clamped values will be masked out)
        safe_positions = jnp.clip(positions, 0, T - 1)  # [T, W]

        # Gather neighbors
        # x_flat[safe_positions] -> [T, W, D]
        windows = x_flat[safe_positions]  # [T, W, D]

        # Zero out invalid positions
        windows = jnp.where(valid[:, :, None], windows, 0.0)

        # If cache is provided, fill in the pre-sequence positions from cache
        if cache is not None:
            # cache: [N, W-1, D]
            # For positions where offset < 0 relative to seq start, use cache
            # Specifically for token t at offset w:
            #   neighbor_pos = t + offsets[w]
            #   if neighbor_pos < seq_starts[t]:
            #     cache_idx = (W-1) + (neighbor_pos - seq_starts[t])
            #     -> (W-1) + offsets[w] + (t - seq_starts[t])
            # Only applicable for first (W-1) tokens of each sequence
            cache_positions_in_cache = (W - 1) + positions - seq_starts[:, None]  # [T, W]
            cache_valid = (~valid) & (cache_positions_in_cache >= 0) & (cache_positions_in_cache < W - 1)
            safe_cache_pos = jnp.clip(cache_positions_in_cache, 0, W - 2)
            seq_cache = cache[seq_ids]  # [T, W-1, D]
            cache_values = seq_cache[jnp.arange(T)[:, None], safe_cache_pos, :]  # [T, W, D]
            windows = jnp.where(cache_valid[:, :, None], cache_values, windows)

        # Depthwise convolution: sum over kernel taps
        # windows: [T, W, D], w_kernel: [W, D] -> y: [T, D]
        y_flat = jnp.sum(windows * w_kernel[None, :, :], axis=1)  # [T, D]

        # Add bias if present
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            y_flat = y_flat + self.conv.bias.value.reshape(D)

        # SiLU activation
        y_flat = jax.nn.silu(y_flat.astype(jnp.float32)).astype(x.dtype)

        y = y_flat[None, :, :]  # [1, T, D]

        # Extract final conv states per sequence
        new_cache = None
        if output_final_state and W > 1:
            # For each sequence n, take last (W-1) tokens as conv state
            # cu_seqlens[n+1] - (W-1) : cu_seqlens[n+1]
            def extract_one_cache(n):
                eos = cu_seqlens[n + 1]
                bos = cu_seqlens[n]
                seq_len = eos - bos
                # Positions to extract: eos - (W-1) .. eos-1
                # But if seq_len < W-1, we need to pad from initial cache
                indices = eos - (W - 1) + jnp.arange(W - 1)
                # valid if >= bos
                valid_mask = indices >= bos
                safe_idx = jnp.clip(indices, 0, T - 1)
                values = x_flat[safe_idx]  # [W-1, D]
                values = jnp.where(valid_mask[:, None], values, 0.0)
                # If initial cache provided and seq is short, fill from cache
                if cache is not None:
                    # cache_offset = (W-1) - seq_len + i  for i = 0..W-2
                    cache_idx = jnp.arange(W - 1) - (seq_len - (W - 1))
                    cache_idx = jnp.clip(cache_idx, 0, W - 2)
                    cache_fill_valid = ~valid_mask
                    cache_vals = cache[n][cache_idx]  # [W-1, D]
                    values = jnp.where(cache_fill_valid[:, None], cache_vals, values)
                return values

            # Static loop (N is typically small)
            new_cache = jax.vmap(extract_one_cache)(jnp.arange(N))  # [N, W-1, D]

        return y, new_cache


# ============================================================================
# FusedRMSNormGated (nnx Module version)
# ============================================================================

class FusedRMSNormGated(nnx.Module):
    """RMSNorm + gating, matching fla FusedRMSNormGated."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        activation: str = "sigmoid",
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ):
        self.activation = activation
        self.dtype = dtype
        self.rms_norm = RMSNorm(
            num_features=dim,
            epsilon=eps,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: Array, gate: Array) -> Array:
        normalized_x = self.rms_norm(x)
        if self.activation == "sigmoid":
            g = jax.nn.sigmoid(gate.astype(jnp.float32))
        elif self.activation in ("silu", "swish"):
            g = jax.nn.silu(gate.astype(jnp.float32))
        else:
            g = gate
        return (normalized_x * g).astype(self.dtype)


# ============================================================================
# Recurrent KDA reference (supports varlen via cu_seqlens)
# ============================================================================

def _recurrent_kda_sequence(
    q: Array, k: Array, v: Array, g: Array, beta: Array,
    initial_state: Array | None,
) -> tuple[Array, Array]:
    """Recurrent KDA for a single sequence.

    Args:
        q,k,v,g: [T, H, D]
        beta: [T, H]
        initial_state: [H, K, V] or None
    Returns:
        o: [T, H, V], final_state: [H, K, V]
    """
    t_len, num_heads, key_dim = q.shape
    value_dim = v.shape[2]

    state0 = jnp.zeros((num_heads, key_dim, value_dim), dtype=jnp.float32)
    if initial_state is not None:
        state0 = state0 + initial_state.astype(jnp.float32)

    def step(carry, inputs):
        state = carry
        q_i, k_i, v_i, g_i, b_i = inputs
        state = state * jnp.exp(g_i)[..., None]
        inner = jnp.sum(k_i[..., None] * state, axis=-2)
        state = state + jnp.einsum(
            "h k, h v -> h k v",
            b_i[..., None] * k_i,
            v_i - inner,
        )
        o_i = jnp.einsum("h k, h k v -> h v", q_i, state)
        return state, o_i

    final_state, o = jax.lax.scan(step, state0, (q, k, v, g, beta), length=t_len)
    return o, final_state


def chunk_kda_reference(
    q: Array, k: Array, v: Array, g: Array, beta: Array,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: Array | None = None,
    **kwargs: Any,
) -> tuple[Array, Array | None]:
    """Reference chunk_kda using recurrent formulation."""
    if use_gate_in_kernel:
        raise ValueError("use_gate_in_kernel not supported in reference")

    q = jnp.asarray(q)
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    g = jnp.asarray(g)
    beta = jnp.asarray(beta)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)

    q = q.astype(jnp.float32) * jnp.float32(scale)
    k = k.astype(jnp.float32)
    v_fp32 = v.astype(jnp.float32)
    g_fp32 = g.astype(jnp.float32)
    beta_fp32 = beta.astype(jnp.float32)

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError("Varlen requires batch size 1")

        total_tokens = q.shape[1]
        num_heads = q.shape[2]
        key_dim = q.shape[3]
        value_dim = v.shape[3]

        q_flat = q[0]
        k_flat = k[0]
        v_flat = v_fp32[0]
        g_flat = g_fp32[0]
        b_flat = beta_fp32[0]

        starts = jnp.zeros((total_tokens,), dtype=jnp.bool_)
        starts = starts.at[cu_seqlens[:-1]].set(True)
        seq_ids = jnp.cumsum(starts.astype(jnp.int32)) - 1

        state_init = jnp.zeros((num_heads, key_dim, value_dim), dtype=jnp.float32)

        def step_varlen(carry, inputs):
            state = carry
            q_i, k_i, v_i, g_i, b_i, is_start_i, seq_id_i = inputs

            if initial_state is not None:
                init_state = initial_state[seq_id_i]
                state = jnp.where(is_start_i, init_state, state)
            else:
                state = jnp.where(is_start_i, 0.0, state)

            state = state * jnp.exp(g_i)[..., None]
            inner = jnp.sum(k_i[..., None] * state, axis=-2)
            state = state + jnp.einsum(
                "h k, h v -> h k v",
                b_i[..., None] * k_i,
                v_i - inner,
            )
            o_i = jnp.einsum("h k, h k v -> h v", q_i, state)
            return state, (o_i, state)

        scan_inputs = (q_flat, k_flat, v_flat, g_flat, b_flat, starts, seq_ids)
        _, (o_flat, states_flat) = jax.lax.scan(step_varlen, state_init, scan_inputs)

        o = o_flat[None, ...]

        if output_final_state:
            end_indices = cu_seqlens[1:] - 1
            final_state = states_flat[end_indices]
        else:
            final_state = None
    else:
        batch = q.shape[0]
        num_heads = q.shape[2]
        key_dim = q.shape[3]
        value_dim = v.shape[3]
        init = initial_state

        def per_batch(qb, kb, vb, gb, bb, init_b):
            return _recurrent_kda_sequence(qb, kb, vb, gb, bb, init_b)

        if init is None:
            init = jnp.zeros((batch, num_heads, key_dim, value_dim), dtype=jnp.float32)

        o, final_state = jax.vmap(per_batch)(q, k, v_fp32, g_fp32, beta_fp32, init)

    if not output_final_state:
        final_state = None
    return o.astype(v.dtype), final_state


def fused_recurrent_kda_reference(
    q: Array, k: Array, v: Array, g: Array, beta: Array,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: Array | None = None,
    **kwargs: Any,
) -> tuple[Array, Array | None]:
    """Reference fused_recurrent_kda using the recurrent formulation."""
    if use_gate_in_kernel:
        raise ValueError("use_gate_in_kernel not supported in reference")
    return chunk_kda_reference(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=False,
        cu_seqlens=cu_seqlens,
        **kwargs,
    )


# ============================================================================
# Cache container
# ============================================================================

class KDACache(NamedTuple):
    """Cache container for KimiDelta Attention."""
    recurrent_state: Array | None
    conv_state: tuple[Array | None, Array | None, Array | None] | None


# ============================================================================
# KimiDeltaAttention (Flax nnx.Module, standalone)
# ============================================================================

class KimiDeltaAttention(nnx.Module):
    """KimiDelta Attention (KDA) - standalone JAX/Flax nnx implementation.

    Aligned with fla/fla/layers/kda.py (PyTorch) for precision comparison.
    Supports varlen (packed sequences) via cu_seqlens.
    Does NOT support GVA (num_v_heads > num_heads).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        head_dim: int = 128,
        num_heads: int = 16,
        mode: str = "chunk",
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        dtype: jnp.dtype = jnp.float32,
        weight_dtype: jnp.dtype = jnp.float32,
        kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        assert expand_v == 1.0, "GVA not supported (expand_v must be 1.0)"
        assert mode in ("chunk", "fused_recurrent")

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * self.head_k_dim
        self.value_dim = num_heads * self.head_v_dim
        self.mode = mode
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.norm_eps = norm_eps
        self.dtype = dtype

        # --- Projections (aligned with fla) ---
        proj_kwargs = dict(dtype=dtype, weight_dtype=weight_dtype,
                           kernel_init=kernel_init, use_bias=False, rngs=rngs)

        self.q_proj = DenseGeneral(
            in_features_shape=hidden_size, out_features_shape=self.key_dim,
            axis=-1, **proj_kwargs)
        self.k_proj = DenseGeneral(
            in_features_shape=hidden_size, out_features_shape=self.key_dim,
            axis=-1, **proj_kwargs)
        self.v_proj = DenseGeneral(
            in_features_shape=hidden_size, out_features_shape=self.value_dim,
            axis=-1, **proj_kwargs)

        # --- Short Convolution ---
        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim, kernel_size=conv_size,
                use_bias=conv_bias, dtype=dtype, rngs=rngs)
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim, kernel_size=conv_size,
                use_bias=conv_bias, dtype=dtype, rngs=rngs)
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim, kernel_size=conv_size,
                use_bias=conv_bias, dtype=dtype, rngs=rngs)

        # --- f_proj (gate/decay): two-layer MLP, no activation, no bias ---
        # fla: nn.Sequential(Linear(hidden, head_v_dim, bias=False),
        #                     Linear(head_v_dim, key_dim, bias=False))
        self.f_a_proj = DenseGeneral(
            in_features_shape=hidden_size, out_features_shape=self.head_v_dim,
            axis=-1, **proj_kwargs)
        self.f_b_proj = DenseGeneral(
            in_features_shape=self.head_v_dim, out_features_shape=self.key_dim,
            axis=-1, **proj_kwargs)

        # --- b_proj: Linear(hidden, num_heads, bias=False) ---
        self.b_proj = DenseGeneral(
            in_features_shape=hidden_size, out_features_shape=num_heads,
            axis=-1, **proj_kwargs)

        # --- A_log, dt_bias ---
        def a_log_init(key, shape):
            return jnp.log(jax.random.uniform(
                key, shape=shape, dtype=jnp.float32, minval=1.0, maxval=16.0))

        self.A_log = nnx.Param(a_log_init(rngs.params(), (num_heads,)))
        self.dt_bias = nnx.Param(jnp.zeros((self.key_dim,), dtype=jnp.float32))

        # --- g_proj (output gate): two-layer MLP ---
        # fla: nn.Sequential(Linear(hidden, head_v_dim, bias=False),
        #                     Linear(head_v_dim, value_dim, bias=True))
        self.g_a_proj = DenseGeneral(
            in_features_shape=hidden_size, out_features_shape=self.head_v_dim,
            axis=-1, **proj_kwargs)
        proj_kwargs_with_bias = dict(proj_kwargs)
        proj_kwargs_with_bias['use_bias'] = True
        self.g_b_proj = DenseGeneral(
            in_features_shape=self.head_v_dim, out_features_shape=self.value_dim,
            axis=-1, **proj_kwargs_with_bias)

        # --- Output norm + proj ---
        self.o_norm = FusedRMSNormGated(
            dim=self.head_v_dim, eps=norm_eps, activation="sigmoid",
            dtype=dtype, rngs=rngs)

        self.o_proj = DenseGeneral(
            in_features_shape=self.value_dim, out_features_shape=hidden_size,
            axis=-1, **proj_kwargs)

    def __call__(
        self,
        hidden_states: Array,
        *,
        cache: KDACache | None = None,
        cu_seqlens: Array | None = None,
        output_final_state: bool = False,
        return_intermediates: bool = False,
        training: bool = False,
    ) -> tuple[Array, KDACache | None, dict[str, Array] | None]:
        """Forward pass aligned with fla KimiDeltaAttention.forward().

        Args:
            hidden_states: [B, T, hidden_size].
                Varlen: [1, TotalT, hidden_size] with cu_seqlens.
            cache: optional KDACache.
            cu_seqlens: [N+1] int32, cumulative sequence lengths.
            output_final_state: whether to output recurrent + conv final states.
            return_intermediates: return dict of all intermediate tensors.
            training: training mode flag.

        Returns:
            (o_out, new_cache, intermediates)
        """
        use_cache = output_final_state or cache is not None
        batch, q_len, _ = hidden_states.shape

        # Mode selection (aligned with fla)
        mode = "fused_recurrent" if (q_len <= 64 and not training) else self.mode
        if training:
            assert mode == "chunk", "Only chunk mode supported in training."

        # Unpack cache
        conv_state_q = conv_state_k = conv_state_v = None
        recurrent_state = None
        if cache is not None:
            recurrent_state = cache.recurrent_state
            if cache.conv_state is not None:
                conv_state_q, conv_state_k, conv_state_v = cache.conv_state

        # === Step 1: Projection + Short Convolution ===
        q_proj_out = self.q_proj(hidden_states)
        k_proj_out = self.k_proj(hidden_states)
        v_proj_out = self.v_proj(hidden_states)

        if self.use_short_conv:
            q_conv_out, conv_state_q = self.q_conv1d(
                x=q_proj_out, cache=conv_state_q,
                output_final_state=use_cache, cu_seqlens=cu_seqlens)
            k_conv_out, conv_state_k = self.k_conv1d(
                x=k_proj_out, cache=conv_state_k,
                output_final_state=use_cache, cu_seqlens=cu_seqlens)
            v_conv_out, conv_state_v = self.v_conv1d(
                x=v_proj_out, cache=conv_state_v,
                output_final_state=use_cache, cu_seqlens=cu_seqlens)
        else:
            q_conv_out = jax.nn.silu(q_proj_out.astype(jnp.float32)).astype(hidden_states.dtype)
            k_conv_out = jax.nn.silu(k_proj_out.astype(jnp.float32)).astype(hidden_states.dtype)
            v_conv_out = jax.nn.silu(v_proj_out.astype(jnp.float32)).astype(hidden_states.dtype)

        # === Step 2: Data-dependent parameters ===
        # f_proj -> g (gate/decay)
        g_pre_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g_pre_gate = g_pre_gate.reshape(batch, q_len, self.num_heads, self.head_k_dim)

        # fused KDA gate
        g_post_gate = fused_kda_gate(g_pre_gate, self.A_log.value, self.dt_bias.value)

        # beta
        beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32))

        # Reshape q, k, v to [B, T, H, D]
        q = q_conv_out.reshape(batch, q_len, self.num_heads, self.head_k_dim)
        k = k_conv_out.reshape(batch, q_len, self.num_heads, self.head_k_dim)
        v = v_conv_out.reshape(batch, q_len, self.num_heads, self.head_v_dim)

        # === Step 3: Core KDA computation (Delta Rule) ===
        if mode == "chunk":
            o_pre_norm, recurrent_state = chunk_kda_reference(
                q=q, k=k, v=v, g=g_post_gate, beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
                cu_seqlens=cu_seqlens,
            )
        else:
            o_pre_norm, recurrent_state = fused_recurrent_kda_reference(
                q=q, k=k, v=v, g=g_post_gate, beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
                cu_seqlens=cu_seqlens,
            )

        # === Step 4: Output norm + projection ===
        g_for_o_norm = self.g_b_proj(self.g_a_proj(hidden_states))
        g_for_o_norm = g_for_o_norm.reshape(batch, q_len, self.num_heads, self.head_v_dim)

        o_post_norm = self.o_norm(o_pre_norm, g_for_o_norm)

        o_out = o_post_norm.reshape(batch, q_len, -1)
        o_out = self.o_proj(o_out)

        # Cache
        new_cache = None
        if use_cache:
            new_cache = KDACache(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
            )

        # Intermediates
        intermediates = None
        if return_intermediates:
            intermediates = {
                "q_proj_out": q_proj_out,
                "k_proj_out": k_proj_out,
                "v_proj_out": v_proj_out,
                "q_conv_out": q_conv_out,
                "k_conv_out": k_conv_out,
                "v_conv_out": v_conv_out,
                "g_pre_gate": g_pre_gate,
                "g_post_gate": g_post_gate,
                "g_for_o_norm": g_for_o_norm,
                "beta": beta,
                "A_log": self.A_log.value,
                "dt_bias": self.dt_bias.value,
                "o_pre_norm": o_pre_norm,
                "o_post_norm": o_post_norm,
                "o_out": o_out,
                "recurrent_state": recurrent_state,
            }

        return o_out, new_cache, intermediates
