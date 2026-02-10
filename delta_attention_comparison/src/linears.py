import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Iterable, Any

from src.common_types import DType, Array
from src.initializers import NdInitializer, nd_dense_init, default_bias_init

def canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)

def normalize_axes(axes: Iterable[int], ndim: int) -> tuple[int, ...]:
  return tuple(ax if ax >= 0 else ndim + ax for ax in axes)

class DenseGeneral(nnx.Module):
  def __init__(
      self,
      in_features_shape: Iterable[int] | int,
      out_features_shape: Iterable[int] | int,
      axis: Iterable[int] | int = -1,
      weight_dtype: DType = jnp.float32,
      dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "truncated_normal"),
      use_bias: bool = False,
      *,
      rngs: nnx.Rngs = None,
      **kwargs, # Ignore extra kwargs like shard_mode, quant, etc.
  ):
    self.in_features_shape = canonicalize_tuple(in_features_shape)
    self.out_features_shape = canonicalize_tuple(out_features_shape)
    self.axis = canonicalize_tuple(axis)
    self.weight_dtype = weight_dtype
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.use_bias = use_bias

    kernel_shape = self.in_features_shape + self.out_features_shape
    kernel_in_axis = np.arange(len(self.axis))
    kernel_out_axis = np.arange(len(self.axis), len(self.axis) + len(self.out_features_shape))

    self.kernel = nnx.Param(
        self.kernel_init(
            rngs.params(),
            kernel_shape,
            self.weight_dtype,
            kernel_in_axis,
            kernel_out_axis,
        )
    )

    if self.use_bias:
      bias_shape = kernel_shape[-len(self.out_features_shape) :]
      self.bias = nnx.Param(
          default_bias_init(rngs.params(), bias_shape, self.weight_dtype)
      )
    else:
      self.bias = None

  def __call__(self, inputs: Array) -> Array:
    inputs = jnp.asarray(inputs, self.dtype)
    norm_axis = normalize_axes(self.axis, inputs.ndim)

    kernel = jnp.asarray(self.kernel.value, self.dtype)

    contract_ind = tuple(range(0, len(self.axis)))
    output = jax.lax.dot_general(
        inputs, kernel, ((norm_axis, contract_ind), ((), ()))
    )

    if self.bias is not None:
      bias = jnp.asarray(self.bias.value, self.dtype)
      output += bias
    return output