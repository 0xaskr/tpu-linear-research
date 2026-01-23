from typing import Any
from flax import nnx
from flax import linen as nn
import jax
import jax.numpy as jnp
from src.common_types import Array, DType

class RMSNorm(nnx.Module):
  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-6,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      scale_init: Any = nn.initializers.ones,
      scale_offset: float = 0.0,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.scale_offset = scale_offset
    self.scale = nnx.Param(
        scale_init(rngs.params(), (num_features,), weight_dtype)
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * jax.lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = jnp.asarray(self.scale.value, self.dtype)
    effective_scale = scale + self.scale_offset
    return y * effective_scale

def l2norm(x: Array, dim: int = -1, eps: float = 1e-6) -> Array:
  inv_norm = jax.lax.rsqrt((x * x).sum(axis=dim, keepdims=True) + jnp.array(eps, dtype=x.dtype))
  return x * inv_norm

class Qwen3NextRMSNormGated(nnx.Module):
  def __init__(self, num_features: int, eps: float, dtype: DType, weight_dtype: DType, *, rngs: nnx.Rngs):
    self.num_features = num_features
    self.eps = eps
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.rms_norm = RMSNorm(
            num_features=num_features,
            epsilon=eps,
            dtype=dtype,
            weight_dtype=weight_dtype,
            scale_init=nn.initializers.ones,
            rngs=rngs,
    )

  def __call__(self, hidden_states: Array, gate: Array) -> Array:
    normalized_states = self.rms_norm(hidden_states)
    gated_states = normalized_states * jax.nn.silu(gate.astype(jnp.float32))
    return gated_states.astype(self.dtype)