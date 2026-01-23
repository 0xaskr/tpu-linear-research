

import torch
from einops import rearrange


# cpu版本算法实现
def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """
    Naive recurrent implementation of KDA (Kimi/Kernel Delta Attention).
    KDA (Kimi/Kernel Delta Attention) 的朴素循环实现。

    This function implements the Delta Rule update step-by-step.
    Ideally, we want to find a matrix S such that S * k = v.
    Using Gradient Descent (SGD) on the objective 0.5 * ||v - S * k||^2, we get the update rule.
    该函数逐步实现了 Delta 规则更新。
    理想情况下, 我们希望找到一个矩阵 S 使得 S * k = v。
    通过对目标函数 0.5 * ||v - S * k||^2 使用梯度下降 (SGD), 我们得到了更新规则。

    Mathematical Formulation (Plain Text):
    数学公式 (纯文本):
    
    Let S_{t-1} be the state at step t-1.
    令 S_{t-1} 为 t-1 步的状态。

    1. Decay the old state:
       衰减旧状态:
       S_{decayed} = S_{t-1} * exp(g_t)
       
       Here, g_t is the log-space decay factor. exp(g_t) acts as the forgetting gate.
       这里, g_t 是对数空间的衰减因子。exp(g_t) 充当遗忘门。

    2. Calculate the estimated value (projection) using current key k_t:
       使用当前键 k_t 计算估计值 (投影):
       v_est = S_{decayed}^T * k_t
       
       Note: In code, S has shape [K, V], so this is effectively (k_t^T * S).
       注意: 在代码中, S 的形状是 [K, V], 所以这实际上是 (k_t^T * S)。

    3. Calculate the residual (error):
       计算残差 (误差):
       error = v_t - v_est
       
       This represents the information in v_t that S cannot yet explain.
       这代表了 v_t 中 S 尚未能解释的信息。

    4. Update the state S using the error and learning rate beta_t:
       使用误差和学习率 beta_t 更新状态 S:
       S_t = S_{decayed} + beta_t * (error * k_t^T)
       
       (In code, outer product of k and error).
       (在代码中, 是 k 和 error 的外积)。

    5. Compute Output:
       计算输出:
       o_t = S_t^T * q_t
       (In code, q_t * S_t).

    Args:
        q (Tensor): Query [B, T, H, K]
        k (Tensor): Key [B, T, H, K]
        v (Tensor): Value [B, T, H, V]
        g (Tensor): Log-decay [B, T, H, K] or broadcastable. 
                    Controls memory retention.
                    对数衰减, 控制记忆保持。
        beta (Tensor): Step size / Learning rate [B, T, H].
                       Determines how much new info overwrites old info.
                       步长/学习率。决定了新信息覆盖旧信息的程度。
        scale (float): Scale for query.
    """
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q * scale

    # S: The memory matrix of shape [B, H, K, V].
    # S: 形状为 [B, H, K, V] 的记忆矩阵。
    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        
        # 1. Decay the state
        # 1. 衰减状态
        S = S * g_i[..., None].exp()
        
        # 2. Compute the residual (v_real - v_predicted)
        # v_predicted = k_i^T * S
        # 2. 计算残差 (真实值 - 预测值)
        # 预测值 = k_i^T * S
        # (k_i[..., None] * S).sum(-2) performs the dot product k^T S
        v_predicted = (k_i[..., None] * S).sum(-2)
        residual = v_i - v_predicted
        
        # 3. Update State using Delta Rule
        # S_new = S_old + beta * k * residual^T
        # 3. 使用 Delta 规则更新状态
        # S_new = S_old + beta * k * residual^T
        # We compute update term: (beta * k) outer_product (residual)
        update = torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, residual)
        S = S + update
        
        # 4. Compute Output
        # o = q * S
        # 4. 计算输出
        # o = q * S
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
        
    if not output_final_state:
        S = None
    return o.to(dtype), S

# cpu版本分块算法实现
def naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    """
    Chunk-wise implementation of KDA.
    KDA 的分块实现。

    This optimizes the recurrent calculation by processing chunks of size `chunk_size` in parallel where possible.
    Inside a chunk, the relationship between inputs and outputs can be modeled as a linear system.
    该方法通过尽可能并行处理大小为 `chunk_size` 的块来优化循环计算。
    在块内部, 输入和输出之间的关系可以被建模为一个线性系统。

    Mechanism (Plain Text):
    机制 (纯文本):
    
    1. Pre-processing:
       Split sequence into chunks.
       Cumulative sum of decays 'g' to handle time-decay efficiently.
       预处理: 将序列分割成块。对衰减 'g' 进行累积求和, 以有效地处理时间衰减。

    2. Intra-Chunk Dependency (Matrix A):
       We need to know how the key at step j affects the state at step i (where i > j) within the same chunk.
       Since S updates are cumulative: S_i = S_{i-1} + update_i
       And update_i depends on (v_i - k_i^T S_{i-1}).
       This creates a dependency chain. We solve this by constructing a transition matrix A.
       
       块内依赖 (矩阵 A):
       我们需要知道同一块内第 j 步的键如何影响第 i 步的状态 (i > j)。
       由于 S 的更新是累积的: S_i = S_{i-1} + update_i
       且 update_i 依赖于 (v_i - k_i^T S_{i-1})。
       这创建了一个依赖链。我们通过构建转移矩阵 A 来解决这个问题。

       A[i, j] essentially represents the influence coefficient of step j on step i.
       It involves the dot product k_i^T * k_j and the beta_j factor.
       A[i, j] 本质上代表了第 j 步对第 i 步的影响系数。
       它涉及点积 k_i^T * k_j 和因子 beta_j。

    3. Solving the Linear System:
       The recurrence S_i = S_{i-1} + beta * k * (v - k^T S_{i-1}) can be rewritten.
       We solve for the "effective" values (u) and "effective" keys (w) that account for intra-chunk updates.
       
       解线性系统:
       循环公式 S_i = S_{i-1} + beta * k * (v - k^T S_{i-1}) 可以重写。
       我们求解“有效”值 (u) 和“有效”键 (w), 它们计入了块内的更新。
       
       The code iteratively solves for A (lower triangular) to account for the chain of updates:
       "I update based on you, you update based on him..."
       代码迭代求解 A (下三角矩阵) 以解释更新链: 
       “我基于你更新, 你基于他更新……”

    4. Inter-Chunk Recurrence:
       Update the global state S from chunk to chunk using the "effective" chunk updates.
       块间循环:
       使用“有效”的块更新, 在块之间更新全局状态 S。

    Args:
        chunk_size (int): Size of the chunk (BT).
    """
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5
    assert T % BT == 0

    # Reshape into chunks [B, H, NT, BT, D]
    # 重塑为块 [B, H, NT, BT, D]
    q, k, v, g, beta = map(lambda x: rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float), [q, k, v, g, beta])
    q = q * scale
    
    # Cumulative decay within chunk
    # 块内的累积衰减
    g = g.cumsum(-2)

    # note that diagonal is masked.
    # 上三角矩阵
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)

    # --- Step 1: Compute Intra-Chunk Interaction Matrix A ---
    # --- 第一步: 计算块内交互矩阵 A ---
    
    # A initial calculation: measures alignment between keys at different steps
    # weighted by decay.
    # A 的初始计算: 测量不同步骤键之间的对齐程度, 并按衰减加权。
    # A[..., i] contains dot products of k_i with all k (in the chunk)
    A = torch.zeros(*q.shape[:-1], BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        # k * (g - g_i).exp(): Decayed keys relative to step i
        # k_i: Current key
        # Einsum: Dot product between current key and historical keys
        # Einsum: 当前键与历史键的点积
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    
    # Apply beta (learning rate)
    # 应用 beta (学习率)
    A = A * beta[..., None]

    # Mask future steps (causal masking)
    # 掩盖未来步骤 (因果掩码)
    A = -A.masked_fill(mask, 0)
    
    # Solve the dependency chain iteratively
    # The update at step i depends on i-1, which depends on i-2...
    # This loop propagates these dependencies through the lower triangular matrix.
    # 迭代解决依赖链
    # 第 i 步的更新依赖于 i-1, i-1 依赖于 i-2...
    # 这个循环通过下三角矩阵传播这些依赖关系。
    for i in range(1, BT):
        # A[..., i, :i]: Row i, columns 0 to i-1 (dependencies on past)
        # Update row i based on weighted sum of previous rows
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    
    # Add Identity matrix and scale by beta again for the final transformation
    # 加上单位矩阵并再次用 beta 缩放, 用于最终变换
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]

    # --- Step 2: Compute Effective Updates (w and u) ---
    # --- 第二步: 计算有效更新 (w 和 u) ---

    # w: Effective keys for block update. Represents how much S changes per unit of S in the past.
    # w: 用于块更新的有效键。表示每单位过去 S 的变化量。
    w = A @ (g.exp() * k)
    
    # u: Effective values for block update.
    # u: 用于块更新的有效值。
    u = A @ v

    # --- Step 3: Recurrent State Update (Inter-Chunk) ---
    # --- 第三步: 循环状态更新 (块间) ---

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    
    # Mask for Query-Key attention within chunk
    # 块内查询-键注意力的掩码
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    
    for i in range(0, NT):
        # Fetch chunk data
        # 获取块数据
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        
        # Calculate Intra-Chunk Attention directly for current output
        # 直接计算当前输出的块内注意力
        # This part accounts for the effect of current chunk's inputs on current chunk's outputs
        # 这部分计算了当前块的输入对当前块输出的影响
        A = torch.zeros(B, H, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j+1, :]
            # Attention scores adjusted by decay
            # 经衰减调整的注意力分数
            A[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
        A = A.masked_fill(mask, 0)
        
        # Calculate residual values for the chunk given the PREVIOUS state S
        # 给定前一个状态 S, 计算该块的残差值
        # v_i = u_i (accumulated values) - w_i @ S (projection of old state)
        v_i = u_i - w_i @ S
        
        # Output = (Contribution from Old State S) + (Contribution from Current Chunk A)
        # 输出 = (来自旧状态 S 的贡献) + (来自当前块 A 的贡献)
        # q_i * S: Standard linear attention from history
        # A @ v_i: Correction from current block
        o[:, :, i] = (q_i * g_i.exp()) @ S + A @ v_i
        
        # Update State S for the NEXT chunk
        # 为下一个块更新状态 S
        # 1. Decay S
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        # 2. Add new info from this chunk
        # k_i (keys) and v_i (residuals)
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
        
    if not output_final_state:
        S = None
    return rearrange(o, 'b h n c d -> b (n c) h d').to(dtype), S

