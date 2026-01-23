# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch
from einops import rearrange

from fla.ops.linear_attn.utils import normalize_output


def naive_recurrent_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float | None = None,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standard Linear Attention 的朴素循环(RNN)实现。
    公式: O_t = (Q_t * S_{t-1})
          S_t = S_{t-1} + K_t^T V_t
    
    这里的思想是将 Attention 从 O(N^2) 降低到 O(N)。
    标准 Attention: Softmax(Q K^T) V
    Linear Attention: Q (K^T V) -> 这里利用结合律先算 K^T V。

    Args:
        q: Query 张量，形状 [Batch, SeqLen, Heads, KeyDim]
        k: Key 张量，形状 [Batch, SeqLen, Heads, KeyDim]
        v: Value 张量，形状 [Batch, SeqLen, Heads, ValDim]
        initial_state: 初始状态 S_0（可选），形状 [Batch, Heads, KeyDim, ValDim]
        output_final_state: 是否输出最后一个时间步的状态 S_T
        scale: 缩放因子，通常是 1 / sqrt(KeyDim)。用于防止点积数值过大。
        normalize: 是否进行分母归一化（类似 Softmax 的分母部分）。对于纯 Linear Attention 通常不需要，除非使用特定的核函数。
    """
    dtype = q.dtype
    # 如果没有提供缩放因子，使用标准的 attention 缩放: 1 / sqrt(d_k)
    if scale is None:
        scale = q.shape[-1] ** -0.5
        
    B, T, H, K, V = *q.shape, v.shape[-1]
    
    # 为了保持数值稳定性，循环过程中累加状态使用 float32
    q, k, v = map(lambda x: x.to(torch.float32), (q, k, v))
    o = torch.empty_like(v)

    # S 是隐状态 (Kv cache)，它是一个矩阵 [KeyDim, ValDim]
    # 相比于 Transformer 需要缓存所有历史 token (SeqLen, KeyDim)，这里只需要缓存一个固定大小的矩阵。
    S = torch.zeros((B, H, K, V), device=q.device, dtype=torch.float32)
    
    if initial_state is not None:
        S = S + initial_state
        
    # 核心循环：时间步 t 从 0 到 T-1
    for t in range(T):
        # 1. 更新状态 S_t
        # S_t = S_{t-1} + k_t^T * v_t
        # einsum 'b h k, b h v -> b h k v': 
        # 对于每个 batch (b) 和 head (h)，计算列向量 k (k x 1) 和行向量 v (1 x v) 的外积，得到 (k x v) 矩阵。
        # 这个外积代表了当前 token t 写入到记忆中的信息。
        S = S + torch.einsum('b h k, b h v -> b h k v', k[:, t], v[:, t])
        
        # 2. 计算输出 O_t
        # O_t = q_t * S_t
        # einsum 'b h k v, b h k -> b h v':
        # 用当前的 query 向量 q (1 x k) 去查询记忆矩阵 S (k x v)。
        # 相当于 Q (K^T V) 的结合律应用。
        o[:, t] = torch.einsum('b h k v, b h k -> b h v', S, q[:, t] * scale)
        
    if normalize:
        o = normalize_output(q * scale, k, o)
        
    return o.to(dtype), S if output_final_state else None


def naive_chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standard Linear Attention 的分块（Chunk-wise）实现。
    这种实现方式在训练时并行度更高，比纯 RNN 循环更快。
    
    核心思想：
    将长序列切分为多个小块 (Chunk)。
    - 块内（Intra-chunk）：使用类似标准 Attention 的并行计算 Q K^T V。因为块很小 (64)，O(L^2) 的代价可以接受。
    - 块间（Inter-chunk）：使用 RNN 的方式传递全局记忆状态 S。
    
    好处：
    既享受了 RNN 的线性复杂度（块间），又享受了 GPU 的并行计算能力（块内）。
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
        
    chunk_size = 64
    
    # 重排形状: [Batch, SeqLen, Heads, Dim] -> [Batch, Heads, NumChunks, ChunkSize, Dim]
    # (n c) = SeqLen, 其中 c 是 chunk_size
    q = rearrange(q, 'b (n c) h d -> b h n c d', c=chunk_size) * scale
    k = rearrange(k, 'b (n c) h d -> b h n c d', c=chunk_size)
    v = rearrange(v, 'b (n c) h d -> b h n c d', c=chunk_size)
    
    # 1. 计算块内的 KV 状态贡献
    # K^T V 在每个块内进行矩阵乘法。
    # shape: [..., ChunkSize, KeyDim]^T @ [..., ChunkSize, ValDim] -> [..., KeyDim, ValDim]
    # 这计算了每个块作为一个整体产生的 "KV 记忆更新"。
    kv = k.transpose(-1, -2) @ v
    
    # 2. 累积块间的状态 (Inter-Chunk)
    # cumsum(2) 沿着块的维度 (n) 进行累加。
    # 这一步相当于 RNN 的部分，算出第 n 个块之前的状态总和 S_{n-1}。
    kv = kv.cumsum(2)
    
    # Shift 操作：
    # 当前块 n 只能看到 *之前* 的块 (0 到 n-1) 累积的状态，不能包含自己。
    # 所以我们在第 0 个位置插入全 0，并把最后一个去掉，实现向右平移一位。
    # kv[:, :, n] 变成了 sum(block_0 ... block_{n-1})
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    
    # 3. 计算 块间注意力 (Inter-Chunk Attention)
    # 用当前块的 Q 查询 *之前所有块* 累积下来的 KV 记忆。
    # Q @ KV_sum
    inter = q @ kv
    
    # 4. 计算 块内注意力 (Intra-Chunk Attention)
    # 这部分类似标准的 Softmax Attention (但没有 Softmax，只有 Feature Map)。
    # 对于块内的每个位置 i, j (i >= j)，直接计算 q_i k_j^T v_j。
    # (Q K^T) 得到 [ChunkSize, ChunkSize] 的分数矩阵。
    intra_scores = q @ k.transpose(-1, -2)
    
    # 应用因果掩码 (Causal Mask)
    # 保证 Q 只能看到当前块内自己之前的位置。
    # triu(..., diagonal=1) 生成上三角掩码（不含对角线），masked_fill 将上三角部分置为 0。
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1)
    intra_scores.masked_fill_(mask, 0)
    
    # 最后乘 V
    intra = intra_scores @ v
    
    # 5. 最终结果 = 块间部分 + 块内部分
    o = inter + intra
    
    if normalize:
        o = normalize_output(q * scale, k, o)
        
    # 恢复形状: [Batch, Heads, NumChunks, ChunkSize, Dim] -> [Batch, SeqLen, Heads, Dim]
    return rearrange(o, 'b h n c d -> b (n c) h d')
