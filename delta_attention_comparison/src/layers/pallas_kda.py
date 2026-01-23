
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools

def solve_unit_lower_triangular(A, b):
    """
    求解 (I + A) x = b 中的 x，其中 A 是严格下三角矩阵。
    使用基于块的前向代入法以在 TPU 上获得更好的性能。
    优化版本：使用 in-place 更新以即减少 Python 开销并促进 XLA 融合。
    
    参数:
        A: VMEM 中的 (N, N) 严格下三角矩阵。
        b: VMEM 中的 (N, D) 矩阵。
        
    返回:
        x: (N, D) 解矩阵。
    """
    N, D = b.shape
    # 向量化更新的块大小
    B = 16 
    num_blocks = N // B
    
    # 直接在 b 上进行操作（XLA 会优化 buffer 复用）
    x = b
    
    for i in range(num_blocks):
        start = i * B
        end = (i + 1) * B
        
        # 1. 逐行求解当前的对角块
        # 提取当前块进行处理
        A_ii = A[start:end, start:end]
        x_block = x[start:end]
        
        # 严格下三角矩阵求逆的对角块内循环
        # 由于 B 很小 (16)，这里的 Python 循环展开是可接受的
        for j in range(B):
            # x_j = b_j - sum_{k<j} A_jk * x_k
            # 只有当 j > 0 时才有依赖
            if j > 0:
                vec = A_ii[j, :j]      # (j,)
                prev = x_block[:j]     # (j, D)
                
                # correction = vec @ prev
                correction = jax.lax.dot_general(
                    vec, prev,
                    (((0,), (0,)), ((), ())),
                    precision=jax.lax.Precision.HIGHEST
                )
                
                # In-place update within the block var
                x_block = x_block.at[j].add(-correction)
        
        # 将求解后的块写回 x
        x = x.at[start:end].set(x_block)
        
        # 2. 使用单个 matmul 更新剩余的行 (Block Forward Substitution)
        if i < num_blocks - 1:
            # 这里的切片操作在 XLA 中会被 lowering 为 slice/dynamic_slice
            
            # A_rest: 下方的 A 块 ((N-end), B)
            A_rest = A[end:, start:end]
            
            # update = A_rest @ x_block
            # (N-end, B) @ (B, D) -> (N-end, D)
            update = jax.lax.dot_general(
                A_rest, x_block,
                (((1,), (0,)), ((), ())),
                precision=jax.lax.Precision.HIGHEST
            )
            
            # 批量更新剩余部分
            x = x.at[end:].add(-update)
            
    return x

def kda_intra_chunk_kernel(
    # 输入 (引用)
    k_ref, g_ref, beta_ref, v_ref,
    # 输出 (引用)
    u_out_ref, w_out_ref,
    # 配置
    chunk_size: int,
    head_dim: int,
):
    # 加载输入到 VMEM
    # k: (C, D), g: (C, D), beta: (C, 1), v: (C, D)
    k = k_ref[0, 0, 0]
    g = g_ref[0, 0, 0]
    beta = beta_ref[0, 0, 0] # (C, 1)
    v = v_ref[0, 0, 0]

    # 1. 计算 A 矩阵
    # A_raw_ij = sum_d k_id * k_jd * exp(g_id - g_jd)
    idx = jnp.arange(chunk_size, dtype=jnp.int32)
    mask = idx[:, None] > idx[None, :]
    
    # 广播 g 到 (C, C, D)
    g_diff = g[:, None, :] - g[None, :, :]
    
    # 使用 einsum 和对数空间掩码以提高效率和稳定性
    # 使用加法掩码以避免 Pallas TPU 中的布尔广播问题 (vector<i1> reshape 问题)
    mask_val = jnp.where(mask, 0.0, -jnp.inf)
    safe_g_diff = g_diff + mask_val[:, :, None]
    
    # 恢复到广播和求和以避免 Pallas 降低复杂 einsum 的问题。
    k_outer = k[:, None, :] * k[None, :, :]
    term = k_outer * jnp.exp(safe_g_diff)
    A_raw = jnp.sum(term, axis=-1)
    
    # 应用 Beta 和 掩码
    # A[i, j] = A_raw[i, j] * beta[i] if i > j else 0
    A = A_raw * beta
    
    # 2. 批量求解 u and w
    # (I + A) u_unscaled = v
    # (I + A) w_unscaled = k * exp(g)
    
    target_w = k * jnp.exp(g)
    # 沿 D 轴合并输入以一起求解: (C, 2D)
    combined_b = jnp.concatenate([v, target_w], axis=-1)
    combined_x = solve_unit_lower_triangular(A, combined_b)
    
    u = combined_x[:, :head_dim] * beta
    w = combined_x[:, head_dim:] * beta
    
    # 存储输出
    u_out_ref[0, 0, 0] = u
    w_out_ref[0, 0, 0] = w

@functools.partial(jax.jit, static_argnames=['chunk_size'])
def kda_intra_chunk_fwd(
    k: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    v: jax.Array,
    chunk_size: int = 128
):
    """
    KDA 块内前向传递的 Pallas 实现。
    
    参数:
        k: (B, H, T, D) 键
        g: (B, H, T, D) 对数衰减的累积和
        beta: (B, H, T) Beta
        v: (B, H, T, D) 值
        chunk_size: Pallas 内核的块大小。
        
    返回:
        u: (B, H, T, D)
        w: (B, H, T, D)
    """
    B, H, T, D = k.shape
    assert T % chunk_size == 0, "序列长度必须能被 chunk_size 整除"
    num_chunks = T // chunk_size
    
    # 重塑以暴露块: (B, H, num_chunks, chunk_size, D)
    k_reshaped = k.reshape(B, H, num_chunks, chunk_size, D)
    g_reshaped = g.reshape(B, H, num_chunks, chunk_size, D)
    beta_reshaped = beta.reshape(B, H, num_chunks, chunk_size, 1)
    v_reshaped = v.reshape(B, H, num_chunks, chunk_size, D)
    
    grid = (B, H, num_chunks)
    
    # 输出缓冲区
    # 可以将输出解释为 (B, H, num_chunks, chunk_size, D)，然后重塑回去
    
    # Pallas 调用
    u_reshaped, w_reshaped = pl.pallas_call(
        functools.partial(kda_intra_chunk_kernel, chunk_size=chunk_size, head_dim=D),
        out_shape=[
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, D), dtype=k.dtype),
            jax.ShapeDtypeStruct(shape=(B, H, num_chunks, chunk_size, D), dtype=k.dtype)
        ],
        in_specs=[
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # k
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # g
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, 1)), # beta
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # v
        ],
        out_specs=[
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # u
            pl.BlockSpec(index_map=lambda i, j, l: (i, j, l, 0, 0), block_shape=(1, 1, 1, chunk_size, D)), # w
        ],
        grid=grid,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel","parallel")),
    )(k_reshaped, g_reshaped, beta_reshaped, v_reshaped)
    
    return u_reshaped.reshape(B, H, T, D), w_reshaped.reshape(B, H, T, D)
