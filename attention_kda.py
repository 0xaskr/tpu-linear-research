# Copyright 2023–2025 Google LLC
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

"""KimiDelta Attention (KDA) reference implementation in MaxText style."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from flax import nnx

from MaxText.common_types import Array, Config, DType
from MaxText.layers import nnx_wrappers
from MaxText.layers.linears import DenseGeneral
from MaxText.layers.normalizations import l2norm


def _get_config_attr(config: Any, name: str, default: Any | None = None) -> Any:
  if hasattr(config, "get"):
    return config.get(name, default)
  return getattr(config, name, default)


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


class KDACache(NamedTuple):
  """Cache container for KimiDelta Attention."""

  recurrent_state: Array | None
  conv_state: tuple[Array | None, Array | None, Array | None] | None


class ShortConvolution(nnx.Module):
  """Depthwise causal 1D convolution with optional cache."""

  def __init__(
      self,
      hidden_size: int,
      kernel_size: int,
      dtype: DType,
      precision: str,
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
        use_bias=False,
        dtype=dtype,
        precision=precision,
        rngs=rngs,
    )

  def __call__(
      self,
      x: Array,
      cache: Array | None = None,
      output_final_state: bool = False,
      cu_seqlens: Array | None = None,
  ) -> tuple[Array, Array | None]:
    if cu_seqlens is not None:
      raise NotImplementedError("cu_seqlens is not supported for ShortConvolution yet.")

    if cache is not None:
      x_full = jnp.concatenate([cache, x], axis=1)
      y_full = self.conv(x_full)
      y = y_full[:, -x.shape[1] :, :]
    else:
      x_full = x
      y = self.conv(x)

    y = jax.nn.silu(y.astype(jnp.float32)).astype(x.dtype)

    new_cache = None
    if output_final_state and self.kernel_size > 1:
      new_cache = x_full[:, -(self.kernel_size - 1) :, :]

    return y, new_cache


def _recurrent_kda_sequence(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    initial_state: Array | None,
) -> tuple[Array, Array]:
  """Recurrent KDA for a single sequence.

  Args:
      q,k,v,g: [T, H, K]
      beta: [T, H]
      initial_state: [H, K, V] or None
  Returns:
      o: [T, H, V]
      final_state: [H, K, V]
  """
  t_len = q.shape[0]
  num_heads = q.shape[1]
  key_dim = q.shape[2]
  value_dim = v.shape[2]

  state0 = jnp.zeros((num_heads, key_dim, value_dim), dtype=jnp.float32)
  if initial_state is not None:
    state0 = state0 + initial_state.astype(jnp.float32)

  def step(
      carry: Array,
      inputs: tuple[Array, Array, Array, Array, Array],
  ) -> tuple[Array, Array]:
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

  inputs = (q, k, v, g, beta)
  final_state, o = jax.lax.scan(step, state0, inputs, length=t_len)
  return o, final_state


def chunk_kda_reference(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    scale: float | None = None,
    initial_state: Array | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: Array | None = None,
    **kwargs: Any,
) -> tuple[Array, Array | None]:
  """Reference chunk_kda using recurrent formulation.

  Returns (o, final_state) aligned with PyTorch chunk_kda outputs.
  """
  if use_gate_in_kernel:
    raise ValueError("use_gate_in_kernel is not supported in reference implementation")

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
      raise ValueError("When cu_seqlens is provided, batch size must be 1")

    total_tokens = q.shape[1]
    num_heads = q.shape[2]
    key_dim = q.shape[3]
    value_dim = v.shape[3]

    q_flat = q[0]
    k_flat = k[0]
    v_flat = v_fp32[0]
    g_flat = g_fp32[0]
    b_flat = beta_fp32[0]

    starts = jnp.zeros((total_tokens,), dtype=bool)
    starts = starts.at[cu_seqlens[:-1]].set(True)
    seq_ids = jnp.cumsum(starts.astype(jnp.int32)) - 1

    state_init = jnp.zeros((num_heads, key_dim, value_dim), dtype=jnp.float32)

    def step_varlen(
        carry: Array,
        inputs: tuple[Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, tuple[Array, Array | None]]:
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

      out_state = state if output_final_state else None
      return state, (o_i, out_state)

    scan_inputs = (q_flat, k_flat, v_flat, g_flat, b_flat, starts, seq_ids)
    _, (o_flat, states_flat) = jax.lax.scan(step_varlen, state_init, scan_inputs)

    o = o_flat[None, ...]

    if output_final_state:
      end_indices = cu_seqlens[1:] - 1
      assert states_flat is not None
      final_state = states_flat[end_indices]
    else:
      final_state = None

  else:
    batch = q.shape[0]
    num_heads = q.shape[2]
    key_dim = q.shape[3]
    value_dim = v.shape[3]
    init = None if initial_state is None else initial_state

    def per_batch(
        qb: Array,
        kb: Array,
        vb: Array,
        gb: Array,
        bb: Array,
        init_b: Array,
    ) -> tuple[Array, Array]:
      return _recurrent_kda_sequence(qb, kb, vb, gb, bb, init_b)

    if init is None:
      init = jnp.zeros((batch, num_heads, key_dim, value_dim), dtype=jnp.float32)
      init = init * 0.0

    o, final_state = jax.vmap(per_batch)(q, k, v_fp32, g_fp32, beta_fp32, init)

  if not output_final_state:
    final_state = None
  return o.astype(v.dtype), final_state


def fused_recurrent_kda_reference(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
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
    raise ValueError("use_gate_in_kernel is not supported in reference implementation")
  return chunk_kda_reference(
      q=q,
      k=k,
      v=v,
      g=g,
      beta=beta,
      scale=scale,
      initial_state=initial_state,
      output_final_state=output_final_state,
      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
      use_gate_in_kernel=False,
      cu_seqlens=cu_seqlens,
      **kwargs,
  )


def kda_as_linen(*, config: Config, name: str | None = None):
  """A factory function to create a KDA layer as a Linen module."""
  return nnx_wrappers.to_linen(
      KimiDeltaAttention,
      config=config,
      name=name,
  )


class KimiDeltaAttention(nnx.Module):
  """KimiDelta Attention (chunk + fused-recurrent reference)."""

  def __init__(
      self,
      config: Config,
      layer_idx: int = 0,
      dtype: DType | None = None,
      weight_dtype: DType | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.layer_idx = layer_idx
    self.dtype = dtype if dtype is not None else _get_config_attr(config, "dtype", jnp.float32)
    self.weight_dtype = weight_dtype if weight_dtype is not None else _get_config_attr(
        config, "weight_dtype", self.dtype
    )
    self.matmul_precision = _get_config_attr(config, "matmul_precision", "default")

    self.hidden_size = _get_config_attr(config, "hidden_size", _get_config_attr(config, "emb_dim"))
    self.head_dim = _get_config_attr(config, "head_dim")
    self.num_heads = _get_config_attr(config, "num_attention_heads", _get_config_attr(config, "num_query_heads"))
    self.conv_size = _get_config_attr(config, "short_conv_kernel_size", 1)
    self.rms_norm_eps = _get_config_attr(config, "rms_norm_eps", 1e-5)
    self.no_kda_lora = _get_config_attr(config, "no_kda_lora", True)

    projection_size = self.head_dim * self.num_heads

    self.q_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=projection_size,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_axes=("embed", "mlp"),
        matmul_precision=self.matmul_precision,
        rngs=rngs,
    )
    self.k_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=projection_size,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_axes=("embed", "mlp"),
        matmul_precision=self.matmul_precision,
        rngs=rngs,
    )
    self.v_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=projection_size,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_axes=("embed", "mlp"),
        matmul_precision=self.matmul_precision,
        rngs=rngs,
    )

    self.q_conv1d = ShortConvolution(
        hidden_size=projection_size,
        kernel_size=self.conv_size,
        dtype=self.dtype,
        precision=self.matmul_precision,
        rngs=rngs,
    )
    self.k_conv1d = ShortConvolution(
        hidden_size=projection_size,
        kernel_size=self.conv_size,
        dtype=self.dtype,
        precision=self.matmul_precision,
        rngs=rngs,
    )
    self.v_conv1d = ShortConvolution(
        hidden_size=projection_size,
        kernel_size=self.conv_size,
        dtype=self.dtype,
        precision=self.matmul_precision,
        rngs=rngs,
    )

    if self.no_kda_lora:
      self.f_a_proj = DenseGeneral(
          in_features_shape=self.hidden_size,
          out_features_shape=projection_size,
          axis=-1,
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_axes=("embed", "mlp"),
          matmul_precision=self.matmul_precision,
          rngs=rngs,
      )
      self.f_b_proj = None
      self.g_a_proj = DenseGeneral(
          in_features_shape=self.hidden_size,
          out_features_shape=projection_size,
          axis=-1,
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_axes=("embed", "mlp"),
          matmul_precision=self.matmul_precision,
          rngs=rngs,
      )
      self.g_b_proj = None
    else:
      self.f_a_proj = DenseGeneral(
          in_features_shape=self.hidden_size,
          out_features_shape=self.head_dim,
          axis=-1,
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_axes=("embed", "mlp"),
          matmul_precision=self.matmul_precision,
          rngs=rngs,
      )
      self.f_b_proj = DenseGeneral(
          in_features_shape=self.head_dim,
          out_features_shape=projection_size,
          axis=-1,
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_axes=("mlp", "mlp"),
          matmul_precision=self.matmul_precision,
          rngs=rngs,
      )
      self.g_a_proj = DenseGeneral(
          in_features_shape=self.hidden_size,
          out_features_shape=self.head_dim,
          axis=-1,
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_axes=("embed", "mlp"),
          matmul_precision=self.matmul_precision,
          rngs=rngs,
      )
      self.g_b_proj = DenseGeneral(
          in_features_shape=self.head_dim,
          out_features_shape=projection_size,
          axis=-1,
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          kernel_axes=("mlp", "mlp"),
          matmul_precision=self.matmul_precision,
          rngs=rngs,
      )

    self.b_proj = DenseGeneral(
        in_features_shape=self.hidden_size,
        out_features_shape=self.num_heads,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_axes=("embed", "heads"),
        matmul_precision=self.matmul_precision,
        rngs=rngs,
    )

    self.A_log = nnx.Param(
        nnx.initializers.uniform(scale=1.0)(rngs.params(), (self.num_heads,), jnp.float32)
    )
    self.dt_bias = nnx.Param(nnx.initializers.normal(stddev=1.0)(rngs.params(), (projection_size,), jnp.float32))
    self.o_norm_scale = nnx.Param(nnx.initializers.ones(rngs.params(), (self.head_dim,), self.weight_dtype))

    self.o_proj = DenseGeneral(
        in_features_shape=projection_size,
        out_features_shape=self.hidden_size,
        axis=-1,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        kernel_axes=("mlp", "embed"),
        matmul_precision=self.matmul_precision,
        rngs=rngs,
    )

  def __call__(
      self,
      hidden_states: Array,
      *,
      cache: KDACache | None = None,
      cu_seqlens: Array | None = None,
      output_final_state: bool = False,
      return_intermediates: bool = False,
      training: bool | None = None,
  ) -> tuple[Array, KDACache | None, dict[str, Array] | None]:
    use_cache = output_final_state or cache is not None
    if training is None:
      training = False

    conv_state_q = conv_state_k = conv_state_v = None
    recurrent_state = None
    if cache is not None:
      recurrent_state = cache.recurrent_state
      if cache.conv_state is not None:
        conv_state_q, conv_state_k, conv_state_v = cache.conv_state

    q_proj_out = self.q_proj(hidden_states)
    k_proj_out = self.k_proj(hidden_states)
    v_proj_out = self.v_proj(hidden_states)

    q_conv_out, conv_state_q = self.q_conv1d(
        x=q_proj_out,
        cache=conv_state_q,
        output_final_state=use_cache,
        cu_seqlens=cu_seqlens,
    )
    k_conv_out, conv_state_k = self.k_conv1d(
        x=k_proj_out,
        cache=conv_state_k,
        output_final_state=use_cache,
        cu_seqlens=cu_seqlens,
    )
    v_conv_out, conv_state_v = self.v_conv1d(
        x=v_proj_out,
        cache=conv_state_v,
        output_final_state=use_cache,
        cu_seqlens=cu_seqlens,
    )

    if self.no_kda_lora:
      g_pre_gate = self.f_a_proj(hidden_states)
    else:
      assert self.f_b_proj is not None
      g_pre_gate = self.f_b_proj(self.f_a_proj(hidden_states))
    g_pre_gate = g_pre_gate.reshape(
        hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim
    )

    g_post_gate = fused_kda_gate(g_pre_gate, self.A_log.value, self.dt_bias.value)

    beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32))

    q = q_conv_out.reshape(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim)
    k = k_conv_out.reshape(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim)
    v = v_conv_out.reshape(hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim)

    seq_len = hidden_states.shape[1]
    mode = "fused_recurrent" if seq_len <= 64 else "chunk"
    if training and mode == "fused_recurrent":
      raise ValueError("Only chunk mode is supported in training.")

    if mode == "chunk":
      o_pre_norm, recurrent_state = chunk_kda_reference(
          q=q,
          k=k,
          v=v,
          g=g_post_gate,
          beta=beta,
          initial_state=recurrent_state,
          output_final_state=use_cache,
          use_qk_l2norm_in_kernel=True,
          use_gate_in_kernel=False,
          cu_seqlens=cu_seqlens,
      )
    else:
      o_pre_norm, recurrent_state = fused_recurrent_kda_reference(
          q=q,
          k=k,
          v=v,
          g=g_post_gate,
          beta=beta,
          initial_state=recurrent_state,
          output_final_state=use_cache,
          use_qk_l2norm_in_kernel=True,
          use_gate_in_kernel=False,
          cu_seqlens=cu_seqlens,
      )

    if self.no_kda_lora:
      g_for_o_norm = self.g_a_proj(hidden_states)
    else:
      assert self.g_b_proj is not None
      g_for_o_norm = self.g_b_proj(self.g_a_proj(hidden_states))
    g_for_o_norm = g_for_o_norm.reshape(
        hidden_states.shape[0], hidden_states.shape[1], self.num_heads, self.head_dim
    )

    o_post_norm = fused_rms_norm_gated(
        o_pre_norm,
        g_for_o_norm,
        weight=self.o_norm_scale.value,
        eps=self.rms_norm_eps,
        activation="sigmoid",
    )

    o_out = o_post_norm.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
    o_out = self.o_proj(o_out)

    new_cache = None
    if use_cache:
      new_cache = KDACache(
          recurrent_state=recurrent_state,
          conv_state=(conv_state_q, conv_state_k, conv_state_v),
      )

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
          "conv_state_q": conv_state_q,
          "conv_state_k": conv_state_k,
          "conv_state_v": conv_state_v,
          "recurrent_state": recurrent_state,
      }

    return o_out, new_cache, intermediates
