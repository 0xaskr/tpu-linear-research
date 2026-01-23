import torch
from einops import repeat


def naive_recurrent_abc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: int | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool | None = False,
) -> torch.Tensor:
    """
    Naive recurrent implementation of ABC (Attention with Bounded Memory Control).
    ABC (Attention with Bounded Memory Control) 的朴素循环实现。
    
    This function implements the chunk-wise or step-wise recurrent form of ABC, which compresses
    history into a fixed number of memory slots.
    该函数实现了 ABC 的分块或逐步循环形式，它将历史信息压缩到固定数量的记忆槽（Memory Slots）中。

    Paper: Attention with Bounded Memory Control
    论文：Attention with Bounded Memory Control

    Dimensions / 维度说明:
    B: Batch size (批次大小)
    H: Number of heads (头数)
    T: Sequence length (序列长度)
    K: Key/Query dimension (键/查询维度)
    V: Value dimension (值维度)
    M: Number of memory slots (记忆槽数量, corresponds to s.shape[-1])

    Mechanism / 机制 (Plain Text):
    The standard attention O = Softmax(Q * K^T) * V is decomposed into two stages using M memory slots.
    标准的注意力机制被分解为使用 M 个记忆槽的两个阶段。
    
    Stage 1: Maintain Key Memory (H_K)
    阶段 1：维护键记忆 (H_K)
    - We write Keys (K) into the M slots weighted by Slot Scores (S).
      我们将键 (K) 写入 M 个槽中，权重由槽分数 (S) 决定。
    - We read from these slots using Query (Q) to get Attention Scores (Ok) over the M slots.
      我们使用查询 (Q) 读取这些槽，得到针对 M 个槽的注意力分数 (Ok)。
    
    Stage 2: Maintain Value Memory (H_V)
    阶段 2：维护值记忆 (H_V)
    - We write Values (V) into the M slots weighted by Slot Scores (S).
      我们将值 (V) 写入 M 个槽中，权重由槽分数 (S) 决定。
    - We read from these slots using the Softmaxed Attention Scores (P) to get the final Output.
      我们使用经过 Softmax 的注意力分数 (P) 读取这些槽，得到最终输出。

    Args:
        q (torch.Tensor): Query tensor of shape [B, H, T, K]. 
                          查询张量。
        k (torch.Tensor): Key tensor of shape [B, H, T, K].
                          键张量。
        v (torch.Tensor): Value tensor of shape [B, H, T, V].
                          值张量。
        s (torch.Tensor): Slot scores tensor of shape [B, H, T, M]. 
                          Controls how much to write to each slot.
                          记忆槽分数组，控制写入每个槽的强度。
        g (torch.Tensor, optional): Gate tensor (decay factor).
                                    If None, calculated from s.
                                    门控张量（衰减因子）。如果为 None，则从 s 计算。
        scale (float, optional): Scaling factor for attention scores. 
                                 Default: K**-0.5.
                                 缩放因子。默认是根号 K 的倒数。
        initial_state (tuple, optional): Initial state for recurrence.
                                         初始循环状态。
        output_final_state (bool): Whether to return the final state.
                                   是否返回最终状态。
    """
    dtype = q.dtype

    # Handle Grouped Query Attention (GQA)
    # 处理分组查询注意力 (GQA)，如果 Q 的头数多于 K/V，需要复制 K/V
    NG = q.shape[1]//k.shape[1]
    
    # Pre-process Gate 'g' if not provided
    # 如果未提供门控 g，则根据 s 计算
    # Logic: normalized log-space difference acts as decay
    if g is None:
        # s is interpreted as log-scores for slots
        # s 被解释为槽的对数分数
        z = s.float().logcumsumexp(2)
        # g effectively normalizes the current step relative to history
        # g 实际上是当前步相对于历史的归一化项
        g = torch.cat((z[:, :, :1], z[:, :, :-1]), 2) - z
        s = torch.exp(s - z)
    
    # Convert to float for numerical stability during accumulation
    # 为了累加过程的数值稳定性，转换为 float 类型
    q, k, v, s, g = map(lambda x: x.float(), (q, k, v, s, g))
    
    # Repeat k, v, s, g for GQA if necessary
    # 如果需要，配合 GQA 复制 k, v, s, g
    k, v, s, g = map(lambda x: repeat(x, 'b h t d -> b (h g) t d', g=NG), (k, v, s, g))
    
    # Handle initial state expansion
    # 处理初始状态的维度扩展
    if initial_state is not None:
        initial_state = tuple(map(lambda x: repeat(x, 'b h k v -> b (h g) k v', g=NG), initial_state))

    # Dimensions
    B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]

    # --- Part 1: Key Memory Recurrence (Calculate Attention Scores) ---
    # --- 第一部分：键记忆循环（计算注意力分数） ---
    
    # hk: Hidden state for Keys. Shape [B, H, K, M].
    # Stores the history of Keys distributed across M slots.
    # hk: 键的隐藏状态。存储了分布在 M 个槽中的键的历史信息。
    hk = torch.zeros(B, H, K, M, dtype=torch.float, device=q.device)
    
    # ok: Output scores from reading memory. Shape [B, H, T, M].
    # ok: 读取记忆得到的输出分数。
    ok = torch.zeros_like(s)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    final_state = None
    if initial_state is not None:
        hk += initial_state[0]

    # Iterate over sequence length
    # 遍历序列长度
    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = s[:, :, i] # Note: In this loop, 'v' role is played by slot scores 's'
                         # 注意：在这个循环中，'value'的角色由槽分数 's' 扮演
        g_i = g[:, :, i].exp()

        # Update Key Memory (hk)
        # Formula: hk_new = hk_old * gate + key * slot_score
        # 公式：新记忆 = 旧记忆 * 门控 + 当前键 * 槽分数
        #
        # k_i[..., None] shape: [B, H, K, 1]
        # v_i[..., None, :] shape (s): [B, H, 1, M]
        # Outer product creates [B, H, K, M] update
        # 外积生成 [B, H, K, M] 的更新量
        hk = hk * g_i[..., None, :] + k_i[..., None] * v_i[..., None, :]
        
        # Read from Key Memory
        # Formula: scores = query * hk
        # 公式：分数 = 查询 * 键记忆
        #
        # q_i[..., None] shape: [B, H, K, 1]
        # hk shape: [B, H, K, M]
        # Sum over K dimension -> [B, H, 1, M] -> squeeze -> [B, H, M]
        # 在 K 维度求和得到针对 M 个槽的注意力分数
        ok[:, :, i] = (q_i[..., None] * hk).sum(-2)

    # Calculate Attention Probabilities (over M slots)
    # 计算注意力概率（在 M 个槽上进行 Softmax）
    qv = ok.softmax(-1)

    # --- Part 2: Value Memory Recurrence (Calculate Output) ---
    # --- 第二部分：值记忆循环（计算输出） ---

    # hv: Hidden state for Values. Shape [B, H, M, V].
    # Stores the history of Values distributed across M slots.
    # hv: 值的隐藏状态。存储了分布在 M 个槽中的值的历史信息。
    hv = torch.zeros(B, H, M, V, dtype=torch.float, device=q.device)
    
    # ov: Final Output. Shape [B, H, T, V].
    # ov: 最终输出。
    ov = torch.zeros_like(v)
    
    if initial_state is not None:
        hv += initial_state[1]

    for i in range(T):
        q_i = qv[:, :, i] # Attention probs act as 'query' here
                          # 注意力概率在这里充当 '查询'
        k_i = s[:, :, i]  # Slot scores act as 'key' for writing values
                          # 槽分数充当写入值的 '键'
        v_i = v[:, :, i]  # Actual values
                          # 实际的值
        g_i = g[:, :, i].exp()

        # Update Value Memory (hv)
        # Formula: hv_new = hv_old * gate + slot_score * value
        # 公式：新记忆 = 旧记忆 * 门控 + 槽分数 * 值
        #
        # k_i[..., None] shape (s): [B, H, M, 1]
        # v_i[..., None, :] shape (v): [B, H, 1, V]
        # Outer product -> [B, H, M, V]
        # 外积生成 [B, H, M, V]
        hv = hv * g_i[..., :, None] + k_i[..., None] * v_i[..., None, :]
        
        # Read from Value Memory
        # Formula: output = attention_probs * hv
        # 公式：输出 = 注意力概率 * 值记忆
        #
        # q_i[..., None] shape (probs): [B, H, M, 1]
        # hv shape: [B, H, M, V]
        # Sum over M dimension -> [B, H, 1, V] -> [B, H, V]
        # 在 M 维度求和（加权平均）得到最终输出
        ov[:, :, i] = (q_i[..., None] * hv).sum(-2)

    if output_final_state:
        final_state = (hk.view(B, -1, NG, K, M)[:, :, 0], hv.view(B, -1, NG, M, V)[:, :, 0])
    
    return ov.to(dtype), final_state


def naive_cumsum_abc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """
    A simple implementation of vanilla ABC (Parallel/Cumsum version).
    ABC 的朴素并行（累积求和）实现。
    
    This implementation aligns with the parallel view where memory slots acts as 
    weighted averages of history.
    该实现对应于并行视图，其中记忆槽充当历史的加权平均值。

    WARNING: For demonstration only. May be numerically unstable.
    警告：仅供演示。可能存在数值不稳定性。

    Mechanism / 机制 (Plain Text):
    1. Normalize slot scores 's' to be positive weights.
       将槽分数 's' 归一化为正权重。
    2. Compute cumulative sum of weights 'z' for normalization.
       计算权重 'z' 的累积和用于归一化。
    3. Compute 'K_slots': Cumulative weighted sum of keys 'k', normalized by 'z'.
       Each slot contains the running average of keys assigned to it.
       计算 'K_slots'：键 'k' 的累积加权和，由 'z' 归一化。每个槽包含分配给它的键的运行平均值。
    4. Compute 'V_slots': Cumulative weighted sum of values 'v', normalized by 'z'.
       Each slot contains the running average of values assigned to it.
       计算 'V_slots'：值 'v' 的累积加权和，同理。
    5. Compute attention 'p' between query 'q' and 'K_slots'.
       计算查询 'q' 和 'K_slots' 之间的注意力 'p'。
    6. Output is weighted sum of 'V_slots' using 'p'.
       输出是 'V_slots' 的加权和，权重为 'p'。
    """

    dtype = q.dtype
    q, k, v, s = map(lambda x: x.float(), (q, k, v, s))

    scale = q.shape[-1] ** -0.5
    
    # [batch_size, n_heads, seq_len, n_slots]
    # Normalize s to be positive (like exp(logits))
    # 将 s 归一化为正数（类似 logits 的指数）
    s = (s - s.max(2, True)[0]).exp()
    
    # Cumulative weight for each slot
    # 每个槽的累积权重
    z = s.cumsum(2)
    
    # [batch_size, n_heads, seq_len, n_slots, d_head]
    # Accumulate K into slots (weighted by s), normalize by total weight z
    # 将 K 累加到槽中（由 s 加权），并用总权重 z 归一化
    K = (s.unsqueeze(-1) * k.unsqueeze(-2)).cumsum(2) / z.unsqueeze(-1)
    
    # Accumulate V into slots (weighted by s), normalize by total weight z
    # 将 V 累加到槽中，同理归一化
    V = (s.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(2) / z.unsqueeze(-1)
    
    # [batch_size, n_heads, seq_len, n_slots]
    # Query the slots: p = softmax(q * K_slots)
    # 查询槽：p = softmax(q * K_slots)
    p = torch.einsum('...d,...md->...m', q * scale, K).softmax(-1)
    
    # [batch_size, n_heads, seq_len, d_head]
    # Read from slots: o = p * V_slots
    # 读取槽：o = p * V_slots
    o = torch.einsum('...m,...md->...d', p, V)
    
    return o.to(dtype), None