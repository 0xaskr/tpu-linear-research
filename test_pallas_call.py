from jax.experimental import pallas as pl
import jax.numpy as jnp
import jax
import functools

B = 16
T = 32
H = 8
K = 64
lhs_shape = [B, T, H, K]
lhs = jnp.arange(jnp.prod(jnp.array(lhs_shape)), dtype=jnp.float32).reshape(lhs_shape)
rhs = jnp.ones([B, T, H, K], dtype=jnp.float32)
# out = jnp.zeros(lhs_shape, dtype=jnp.float32)


# @jax.jit
def pa_kernel(lhs : jax.Array, rhs : jax.Array, out : jax.Array,
              is_debug_mode = True):
    x_i = pl.program_id(0)
    y_i = pl.program_id(1)
    a = lhs[...]
    b = rhs[...]
    c = out[...]
    is_out_block = jnp.logical_and(x_i == 0, y_i == 1)
    
    def true_void(a, c):
        c[...] = a[...]
        return 0

    def false_void(a, c):
        return 0
    
    if (is_debug_mode):
        jax.lax.cond(is_out_block, true_void, false_void, lhs, out)
    # pl.store()
    
    print("lhs reshape", lhs.shape)
    # jax.debug.print("lhs = ", lhs[0,0,0,0])
    # jax.debug.print("xy = {}, lhs shape: {}", (x_i, y_i), lhs.shape)
    # jax.debug.print("x_i: {}, y_i: {}, a: {}, b: {}", x_i, y_i, a[0,0], b[0,0])

lhs_blockSpec = pl.BlockSpec(lhs.shape, 
                             lambda x, y: (0, 0, 0, 0))
rhs_blockSpec = pl.BlockSpec(lhs.shape, 
                             lambda x, y: (0, 0, 0, 0))
out_shape = jax.ShapeDtypeStruct(lhs_shape, dtype=lhs.dtype)
out = pl.pallas_call(
    functools.partial(pa_kernel, is_debug_mode = False),
    out_shape=[out_shape], grid=(2, 2),
               in_specs = [
                   lhs_blockSpec,
                   rhs_blockSpec
               ],
               out_specs=[
                   lhs_blockSpec
               ],
               interpret=False)(lhs, rhs)
print(type(out[0]))
print(out[0].reshape(-1)[:10])





