from typing import Callable
import jax
from flax import linen as nn
import jax.numpy as jnp

# Type aliases
PRNGKey = jnp.ndarray
Shape = tuple[int, ...]
DType = jnp.dtype
Array = jnp.ndarray
Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = int | tuple[int, ...]
NdInitializer = Callable[[PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

default_bias_init = jax.nn.initializers.constant(0.0)

def nd_dense_init(scale, mode, distribution):
  def init_fn(key, shape, dtype, in_axis=-2, out_axis=-1):
    fn = jax.nn.initializers.variance_scaling(scale, mode, distribution, in_axis, out_axis)
    return fn(key, shape, dtype)
  return init_fn