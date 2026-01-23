# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Related files are modified and supported by the Moonshot AI Team

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.kda.chunk_bwd import chunk_kda_bwd_dAv, chunk_kda_bwd_wy_dqkg_fused
from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra, chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_bwd, kda_gate_chunk_cumsum
from fla.ops.kda.wy_fast import recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    disable_recompute: bool = False
):
    # qg = None if disable_recompute is False
    # 1. Intra-chunk computation / 块内计算
    # Calculate local state-space parameters and attention scores within each chunk.
    # 计算每个 chunk 内部的状态空间参数和注意力分数。
    # w, u: pre-computed variables for state updates / 用于状态更新的预计算变量
    # Aqk, Akk: local attention scores / 局部注意力分数
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute
    )
    # 2. Inter-chunk Recurrence / 块间循环
    # Update the recurrent state across chunks using the Delta Rule.
    # 使用 Delta Rule 跨 chunk 更新循环状态 (RNN state)。
    # h: history state after each chunk / 每个 chunk 之后的历史状态
    # v_new: updated values after applying the delta rule / 应用 delta rule 更新后的 value (v - correction)
    
    print("kg_shape = ", kg.shape)  # [2, 128, 4, 64]
    print("k.shape = ", k.shape)    # [2, 128, 4, 64]
    print("w.shape = ", w.shape)    # [2, 128, 4, 64]
    print("u.shape = ", u.shape)    # [2, 128, 4, 64]
    print("g.shape = ", g.shape)    # [2, 128, 4, 64]
    print("initial_state shape = ", initial_state.shape if initial_state is not None else None) # None
    print("output_final_state = ", output_final_state)  # false
    print("cu_seqlens = ", cu_seqlens)  # None
    print("chink_indices = ", chunk_indices)  # None
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    print("h.shape = ", h.shape)
    print("v_new shape = ", v_new.shape)
    print("final_state shape = ", final_state.shape if final_state is not None else None)


    # 3. Output Computation / 输出计算
    # Compute the final output by combining local attention and historical state.
    # 结合局部注意力和历史状态计算最终输出。
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new, # Use the updated values
        g=g,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    if not disable_recompute:
        # Delete to save memory
        w, u, qg, kg, v_new, h = None, None, None, None, None, None

    return o, Aqk, Akk, final_state, w, u, qg, kg, v_new, h


def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    disable_recompute: bool = False,
    **kwargs,
):
    if not disable_recompute:
        # Recompute forward pass intermediates to save memory during training
        # 在反向传播时重计算前向传播的中间变量，以节省显存 (Gradient Checkpointing)
        # w = Akk @ (k * beta)
        # u = Akk @ (v * beta)
        w, u, qg, kg = recompute_w_u_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=Akk,
            gk=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        # Recompute hidden states
        # 重计算隐状态
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            k=kg,
            w=w,
            u=u,
            gk=g,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
        )
    else:
        w, u, qg, kg, v_new, h = kwargs["w"], kwargs["u"], kwargs["qg"], kwargs["kg"], kwargs["v_new"], kwargs["h"]
    
    # 1. Gradient of Attention Scores / 注意力分数的梯度
    # dAqk = do @ v.T
    # dv = A @ do
    dAqk, dv = chunk_kda_bwd_dAv(
        q=q,
        k=k,
        v=v_new,
        do=do,
        A=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    # 2. Gradient of Recurrent State / 循环状态的梯度
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg,
        k=kg,
        w=w,
        gk=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    
    # 3. Gradient of Inputs (Fused) / 输入及参数的梯度 (融合计算)
    dq, dk, dv, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=Akk,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    # 4. Intra-chunk Gradients / 块内梯度处理
    dq, dk, db, dg = chunk_kda_bwd_intra(
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate
    )
    return dq, dk, dv, db, dg, dh0


class ChunkKDAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        disable_recompute: bool = False
    ):
        chunk_size = 64
        g_org = None
        # Prepare chunk indices for variable length sequences
        # 为变长序列准备 chunk 索引
        chunk_indices = prepare_chunk_indices(
            cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None
        # 每个batch 切分成chunk_size 大小的多个chunk块，向上对齐
        
        # 1. Gate Preprocessing / 门控预处理
        # If use_gate_in_kernel is True, compute the cumulative sum of gates (log-space decay).
        # 如果启用 use_gate_in_kernel，计算门控的累积和 (对数空间的衰减)。
        # jax 可能要区分出多个kernel出来
        if use_gate_in_kernel:
            g_org = g
            if safe_gate:
                assert lower_bound is not None, "lower_bound must be set when use safe_gate"
            # 数学操作：Log-Space Cumulative Decay Calculation
            # 1. 瞬时衰减值 (Log-Space): log_decay_t = -exp(A) * softplus(g_t + bias)
            #    这一步将输入的 g (原始分数) 映射为对数域的衰减率。
            # 2. 累积衰减值 (CumSum): G_t = \sum_{i=start}^{t} log_decay_i
            #    物理意义：计算从当前块起始位置到 t 时刻的总衰减量。
            #    在后续使用时，任意两点 i, j 之间的衰减可以通过 G_j - G_i 快速算出。
            g = kda_gate_chunk_cumsum(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=RCP_LN2, # 缩放因子：1/ln(2)，可能是为了适配 exp2 指令
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=lower_bound,
            )
        else:
            # Otherwise, just compute local cumulative sum. / 否则，仅计算局部累积和。
            # 分支逻辑：
            # 如果 `use_gate_in_kernel=False`，说明传进来的 `g` 已经是 "log-space decay" 了。
            # 即 g_raw 已经通过外部 PyTorch 代码变成了 log_decay = softplus(...) 之类的值。
            #
            # 在这种情况下，我们只需要做两件事：
            # 1. 缩放 (Scale): g = g * RCP_LN2
            # 2. 累积 (CumSum): G_t = \sum g_i
            #
            # 公式：
            # g_new[t] = \frac{1}{\ln 2} \sum_{i=\text{start}}^{t} g_{\text{input}}[i]
            g = chunk_local_cumsum(
                g=g,
                scale=RCP_LN2, # RCP_LN2 = 1.0 / ln(2)
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices
            )
        
        # 2. Q/K Normalization / Q/K 归一化
        # Optional L2 normalization for stability. / 可选的 L2 归一化以提高稳定性。
        # 公式 (L2 Norm):
        # x_norm = x / (sqrt(sum(x^2, dim=-1)) + eps)
        # 即：将 Query 和 Key 向量除以它们自身的 L2 范数（模长），使其成为单位向量。
        # 这种操作常用于提升训练稳定性，防止 Attention Score 过大。
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        # 3. Core Forward Pass / 核心前向计算
        (o, Aqk, Akk, final_state, w, u, qg, kg, v_new, h) = chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            safe_gate=safe_gate,
            disable_recompute=disable_recompute
        )

        if disable_recompute is False and use_gate_in_kernel:
            g = None  # type: ignore
        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, g_org, beta, A_log, dt_bias, Aqk, Akk,
            w, u, qg, kg, v_new, h,
            initial_state, cu_seqlens, chunk_indices
        )
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.disable_recompute = disable_recompute
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        (q, q_rstd, k, k_rstd, v, g, g_org, beta, A_log, dt_bias, Aqk, Akk,
         w, u, qg, kg, v_new, h,
         initial_state, cu_seqlens, chunk_indices) = (
            ctx.saved_tensors
        )
        if ctx.disable_recompute is False and ctx.use_gate_in_kernel:
            # Recompute gate values if they were not saved
            # 若未保存，重计算门控值
            g = kda_gate_chunk_cumsum(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=RCP_LN2,
                chunk_size=ctx.chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                lower_bound=ctx.lower_bound
            )
        
        # 1. Main Backward Pass / 核心反向传播
        dq, dk, dv, db, dg, dh0 = chunk_kda_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=ctx.chunk_size,
            safe_gate=ctx.safe_gate,
            disable_recompute=ctx.disable_recompute,
            w=w, u=u, qg=qg, kg=kg, v_new=v_new, h=h
        )
        
        # 2. Gradient for L2 Norm (if used) / L2 归一化的梯度 (若使用)
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        dA, dbias = None, None

        # 3. Gradient for Gate / 门控的梯度
        if ctx.use_gate_in_kernel:
            dg = chunk_local_cumsum(
                dg,
                chunk_size=ctx.chunk_size,
                reverse=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
            # Propagate gradients through the gate function
            # 将梯度反向传播通过门控函数
            dg, dA, dbias = kda_gate_bwd(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                dyg=dg,
                lower_bound=ctx.lower_bound
            )
        else:
            dg = chunk_local_cumsum(
                dg,
                chunk_size=ctx.chunk_size,
                reverse=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
        return (dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), dA, dbias, None, dh0,
                None, None, None, None, None, None, None, None)


@torch.compiler.disable
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    **kwargs,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the KDA attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
            KDA 注意力分数的缩放因子。
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
            初始状态张量。
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
            是否输出最终状态。
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q,k tensor internally. Default: `False`.
            是否在 kernel 内部对 q, k 进行 L2 归一化。
        use_gate_in_kernel (bool):
            Whether to compute the log-space KDA decay internally.
            - If `True`:
              The passed `g` acts as the raw input for `-exp(A_log).view(H, -1) * softplus(g + dt_bias.view(H, K))`.
              Note that as part of the input arguments,
              `A_log` (shape `[H]`) and the optional `dt_bias` (shape `[H * K]`) should be provided.
            - If `False`, `g` is expected to be the pre-computed decay value.
            Default: `False`.
            是否在 kernel 内部计算 KDA 的衰减 (log空间)。
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
            变长序列的累积长度。
        cu_seqlens_cpu (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        safe_gate (bool):
            Whether the kernel can assume the input gate values `g` are in a safe range.
            When `True`, the kernel can use M=16 TensorCore acceleration.
            The safe range is approximately [-5, 0). Default: `False`.
            是否假设输入 gate 在安全范围内 ([-5, 0))。
        lower_bound (Optional[float]):
            Lower bound for the forget gate activation function when `use_gate_in_kernel=True`.
            This parameter modifies the internal forget gate activation and is recommended
            to be set to `-5` when `safe_gate` is enabled. Default: `None`.
        disable_recompute (bool):
            Whether to disable gradient recomputation in the kernel. When `True`, the kernel
            will save all intermediate activations for backward pass, which is beneficial
            for training small models at the cost of increased memory usage. Default: `False`.
            是否禁用梯度重计算。禁用会增加显存占用，但可能对训练小模型有益。

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.kda import chunk_kda
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda')
        >>> g = torch.rand(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> A_log = torch.randn(H, dtype=torch.float32, device='cuda')
        >>> dt_bias = torch.randn(H * K, dtype=torch.float32, device='cuda')
        >>> o, ht = chunk_kda(
            q, k, v, g, beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_kda(
            q, k, v, g, beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")
        if safe_gate:
            if lower_bound is None:
                raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
            if not (-5 <= lower_bound < 0):
                raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")

    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert k.shape[-1] <= 256, "Currently we only support key headdim <=256 for KDA :-("
    assert beta.shape == q.shape[:3], "beta must be of shape (batch size, seq len, num of head)."
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (batch size, seq len, num of head, head dim)."
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        cu_seqlens,
        cu_seqlens_cpu,
        safe_gate,
        lower_bound,
        disable_recompute
    )
    return o, final_state
