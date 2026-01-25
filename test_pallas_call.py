from jax.experimental import pallas as pl
import jax.numpy as jnp
import jax


B = 16
T = 32
H = 8
K = 64
lhs_shape = [B, T, H, K]
lhs = jnp.arange(jnp.prod(jnp.array(lhs_shape)), dtype=jnp.float32).reshape(lhs_shape)
rhs = jnp.ones([B, T, H, K], dtype=jnp.float32)

@jax.jit
def pa_kernel(lhs : jax.Array, rhs : jax.Array, out : jax.Array):
    x_i = pl.program_id(0)
    y_i = pl.program_id(1)
    a = lhs[:,:]
    b = rhs[:,:]
    jax.debug.print("xy = {}, lhs shape: {}, value = {}", (x_i, y_i), lhs.shape, lhs.reshape(-1)[:10])
    # jax.debug.print("x_i: {}, y_i: {}, a: {}, b: {}", x_i, y_i, a[0,0], b[0,0])

lhs_blockSpec = pl.BlockSpec((lhs.shape[0], lhs.shape[1], lhs.shape[2] // 2, lhs_shape[3] // 2), 
                             lambda x, y: (0, 0, x * lhs.shape[2]// 2, y * lhs.shape[3] // 2))
rhs_blockSpec = pl.BlockSpec((lhs.shape[0], lhs.shape[1], lhs.shape[2], lhs_shape[3] // 2), lambda x, y: (0, 0, 0, y * lhs.shape[3] // 2))
out_shape = jax.ShapeDtypeStruct(lhs_shape, dtype=lhs.dtype)
pl.pallas_call(pa_kernel, out_shape=out_shape, grid=(2, 2),
               in_specs = [
                   lhs_blockSpec,
                   rhs_blockSpec
               ],
               interpret=True)(lhs, rhs)
# print(lhs.shape)





