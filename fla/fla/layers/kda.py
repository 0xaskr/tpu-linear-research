# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class KimiDeltaAttention(nn.Module):
    """
    Kimi Delta Attention (KDA) layer implementation.
    Kimi Delta Attention (KDA) 层实现。

    KDA 是一种高效的序列建模层，它结合 了以下特性：
    1. 线性注意力/RNN 机制：通过状态空间模型处理长序列，计算复杂度随序列长度线性增长。
    2. 短卷积 (Short Convolution)：在 Q/K/V 投影后应用局部卷积，增强局部依赖捕获能力。
    3. 双模式运行：训练时使用 Chunk 模式（并行化），推理时使用 Recurrent 模式（低延迟）。
    4. 门控机制：通过数据依赖的门控和衰减参数控制记忆的更新和遗忘。

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
            输入张量的隐藏层维度。
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
            Value 维度的扩展比例。Value 投影的维度可以是 Key 维度的倍数。
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
            每个注意力头(Head)的特征维度。
        num_heads (int, Optional):
            The number of heads. Default: 16.
            注意力头的数量。
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA (Grouped Value Attention) is applied if `num_v_heads` > `num_heads`. Default: `None`.
            Value 投影的头数量。如果大于 num_heads，则启用分组 Value 注意力（类似 GQA 的反向逻辑）。
        mode (str, Optional):
            Which Kimi Delta Attention kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
            计算模式选择：
            - `chunk`: 训练模式，将序列分块并行计算，利用 GPU Tensor Core 加速。
            - `fused_recurrent`: 推理模式，类似 RNN 逐步计算，节省显存并降低延迟。
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
            是否使用短卷积。在进入 Attention 之前对 Q/K/V 进行局部混合，类似于 Mamba/RWKV 的设计。
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference:
            [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
            是否允许负特征值。如论文所述，这允许状态更新产生振荡动力学，可能增强模型的表达能力。
            开启时，beta参数会乘 2。
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
            短卷积的卷积核大小 (kernel size)。
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
            短卷积是否使用偏置项。
        layer_idx (int, Optional):
            The index of the layer. Default: None.
            当前层在模型中的索引，用于 KV Cache 的管理。
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
            输出归一化层 (RMSNorm) 的 epsilon 值。
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> KimiDeltaAttention:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        # 默认 num_v_heads 与 num_heads 相等
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        # Key 的头维度
        self.head_k_dim = head_dim
        # Value 的头维度 (根据扩展比例)
        self.head_v_dim = int(self.head_dim * self.expand_v)

        # 投影层的总输出维度 = 头数 * 头维度
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # Consistency check: Ensure expand_v produces integer values
        # 维度一致性检查：确保扩展后的维度是整数
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )

        # 检查头数的倍数关系
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )

        # 检查 Head 维度计算
        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        # =============== 模型层定义 ===============

        # 1. 基础线性投影层 (W_q, W_k, W_v)
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # 2. 短卷积 (Short Convolution) 组件
        # 在时间维度上进行类似于 1D-Conv 的操作，提供局部的 token 混合能力
        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu", # SiLU (Swish) 激活函数
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        # 3. 门控和参数生成网络
        # f_proj: 生成 'g' (通常作为衰减 decay 或门控 gate 使用)。
        # 结构是 dim -> head_v_dim -> key_dim 的两层 MLP (无激活?)，这里有些特殊
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        # b_proj: 生成 'beta'，用于控制更新步长/遗忘强度。
        # 输出维度为 num_heads，即每个头有一个独立的 beta。
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        # 4. 可学习的状态动力学参数
        # A_log: 状态转移矩阵 A 的对数参数化。初始化为 [1, 16] 之间的值。
        # 这些参数决定了记忆的保留时长。
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        # dt_bias: Delta Time 偏置，用于微调时间步长。
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True

        # 5. 输出门控和归一化
        # g_proj (第二个): 生成用于最终输出 FusedRMSNormGated 的门控信号。
        # 注意区分这个 g_proj 和上面的 f_proj。
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        # o_norm: 类似于 RMSNorm，但带有门控机制 (Gated RMSNorm)，有助于稳定训练。
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        # o_proj: 最终的输出线性投影 W_o。
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        # attention_mask (注意力掩码): 在 KDA 中主要用于处理 Padding。
        # 1. 作用: 标识有效 Token (1) 和填充 Token (0)。
        # 2. 形状限制: 只接受 [Batch, Seq] 的 2D 掩码。
        #    - 不同于标准 Attention 的 [B, S, S] 矩阵，KDA 是递归模型 (RNN)，天生就是 Causal (因果) 的，
        #      不需要 Mask 来“遮挡未来”。因此这里只用来区分 Padding。
        # 数据完整性检查：Attention Mask 必须是 2D [Batch, Seq]
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
        print("attention mask shape", attention_mask.shape if attention_mask is not None else None)
        # hidden_states: [batch_size, seq_len, hidden_size]

        # hidden states: [batch_size, seq_len, hidden_size]
        # batch_size (B): 批大小。指有多少个独立的句子/样本正在被同时并行处理。
        #                 例如 batch_size=32 意味着显卡同时在算 32 个不同的用户请求。
        #                 并不是指“一句话有多少Token”，Token数是由 seq_len (q_len) 决定的。
        # q_len (S/L): 序列长度。实际上是指当前输入的长度。
        #              问：不同样本会有不同的 seq_len 吗？
        #              答：在物理 Tensor 层面不行 (必须对齐到统一长度才能打包成 Tensor 矩阵)。
        #                  但逻辑上是的。现实中句子长短不一，短句子会补 0 (Padding) 到和长句子一样长。
        #                  这就是为什么我们需要 attention_mask 来标记“哪些是真 token，哪些是凑数的 0”。
        batch_size, q_len, _ = hidden_states.shape

        # 智能模式切换：
        # 如果序列很短 (<=64) 且在推理模式 (not training)，强制使用 recurrent 以提高小 batch/短序列效率。
        # change to inference mode.
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        # 训练模式下强制使用 chunk 模式
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        # 获取上一时刻的状态（如果是自回归生成）
        # past_key_values: 用于推理时的 KV Cache。
        # 不同于 Transformer 缓存所有的 Key/Value，KDA 这类 RNN 模型只需要缓存当前时刻的：
        # 1. Recurrent State (RNN 隐藏状态，总结了所有历史信息)
        # 2. Conv State (短卷积状态，保存了最近几个 Token 用于局部卷积)
        # 这样可以将显存占用从 O(N) 降低到 O(1)。
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # 处理 Padding 和 变长序列 (VarLen)
        # VarLen (Variable Length / Packed Sequences): 是一种针对变长序列的高效计算机制。
        # 1. 传统方式 (Padding): 为了组成 Batch tensor，短句子必须补 0 (Padding)。这些 0 也会参与矩阵运算，浪费显存和算力。
        # 2. VarLen 方式 (Unpadding): 将 Batch 内所有句子的有效 Token 拼接成一个一维长序列 (Total_Tokens, Dim)。
        #    并使用 cu_seqlens (Cumulative Sequence Lengths) 数组来标记每个句子的起止位置 (如 [0, 10, 25] 表示第一句长度10，第二句长度15)。
        #    KDA Kernel 会根据 cu_seqlens 自动在句子边界重置记忆状态，防止信息跨句子泄漏。
        cu_seqlens = kwargs.get("cu_seqlens")
        if attention_mask is not None:
            print("q_len = ", q_len)
            print("new attention mask:", attention_mask[:, -q_len:].shape)
            # get_unpad_data 用于移除 padding，将 batch 维度展平，以便高效计算
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            # 合并batch 和 seq, 变成 (Total_Tokens, Dim), 然后拿来做gather, 这里其实可以用TPU做

            # 为何使用 attention_mask[:, -q_len:] ?
            # 因为在推理 (Inference) 的解码阶段，q_len 可能只有 1 (生成一个 token)。
            # 但 attention_mask 传入的往往是完整的历史长度 (cache_len + 1)。
            # 我们只需要当前这一步 (最后 q_len 个时间步) 的 mask 信息来判断这一步是否是 padding。

        # === 步骤 1: 投影与局部混合 (Projection & Short Convolution) ===
        # "先看一眼上下文"：在进入核心记忆模块前，通过短卷积混合相邻 Token 的信息。
        # 这一步负责捕捉局部特征（如词组搭配），让后续的线性 Attention 也能感知局部语境。
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            # 从 Cache 中恢复卷积状态 (注意：卷积状态 Conv State 不同于 RNN 的记忆状态 Recurrent State)
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

            # 对 Q/K/V 分别进行： 线性投影 -> 卷积处理 -> 激活
            # Q (Query): 我们想要查询什么信息？
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            # K (Key): 这条信息的索引/标签是什么？（用于检索）
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            # V (Value): 这条信息的具体内容是什么？（需要被记住的内容）
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            # 如果不使用卷积，则直接投影并使用 SiLU 激活
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        # q: [batch_size, seq_len, key_dim]
        # k: [batch_size, seq_len, key_dim]
        # v: [batch_size, seq_len, value_dim]

        # === 步骤 2: 生成数据依赖的参数 (Data-dependent parameters) ===
        # 这也是“选择性记忆”的关键。模型根据当前输入，动态决定：
        # 1. g (Gate/Decay): 遗忘门。决定这个时刻的记忆保留多久。"这件事能记多久？"
        g = self.f_proj(hidden_states)
        # 2. beta (Step Size): 更新步长。决定这个时刻的信息有多重要，需要多大程度修正现有记忆。"这件事有多重要？"
        # 使用 Sigmoid 限制在 (0, 1) 之间。
        beta = self.b_proj(hidden_states).sigmoid()

        # g: [batch_size, seq_len, key_dim]
        # beta: [batch_size, seq_len, num_heads]

        # 重塑张量形状：[..., total_head_dim] -> [..., num_heads, head_dim]
        q, k, g = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, g))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        # q/k/g: [batch_size, seq_len, num_heads, head_dim]
        # v: [batch_size, seq_len, num_v_heads, head_v_dim]

        # 处理多 Value 头 (GVA) 情况：
        # 如果 Value 头数多于 Q/K 头数，需要重复 Q, K, g 来匹配
        # for multi-value attention, we repeat the inputs for simplicity.
        if self.num_v_heads > self.num_heads:
            q, k, g = (repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads) for x in (q, k, g))
            beta = repeat(beta, "... h -> ... (h g)", g=self.num_v_heads // self.num_heads)
        # broadcast

        # 负特征值机制：放大 beta，可能引入振荡动态
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # 获取 RNN 循环状态
        recurrent_state = last_state["recurrent_state"] if last_state is not None else None

        # === 步骤 3: 核心 KDA 计算 (Core Kernel: The Delta Rule) ===
        # 这里的核心逻辑是 Delta Rule (增量规则)： New_State = Old_State + Beta * (V - Old_State * K)
        # 即：只记录当前记忆预测产生的"误差" (Delta)，而不是简单累加信息。
        if mode == "chunk":
            # Chunk 模式：通常用于 Training。在序列长度维度上分块，并行计算块内 attention 和块间 recurrence。
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g, # 门控信号
                beta=beta, # 步长信号
                A_log=self.A_log, # 状态转移参数
                dt_bias=self.dt_bias, # 时间偏差
                initial_state=recurrent_state, # 初始状态
                output_final_state=use_cache, # 是否输出最终状态用于 cache
                use_qk_l2norm_in_kernel=True, # 在 kernel 内部应用 QK L2 Norm 稳定数值
                use_gate_in_kernel=True, # 启用内置门控
                cu_seqlens=cu_seqlens, # 变长序列长度信息
            ) # type: ignore
        elif mode == "fused_recurrent":
            # Recurrent 模式：通常用于 Inference。逐个 token 更新状态，类似传统的 RNN。
            # 先计算 fused gate
            g = fused_kda_gate(g=g, A_log=self.A_log, dt_bias=self.dt_bias)
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # o: [batch_size, seq_len, num_heads, head_v_dim]

        # 更新 KV Cache (包含 recurrent state 和 conv state)
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        # === 步骤 4: 输出归一化与投影 (Output Norm & Proj) ===
        # 这里的归一化也是带门控的 (Gated)，不仅把输出理顺 (Norm)，
        # 还根据输入再次决定放行多少信息 (Gate)。

        # 计算输出门控信号 (来自第二个 g_proj)
        gate_output = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
        # gate_output: [batch_size, seq_len, num_heads, head_v_dim]

        # 融合了门控的 RMSNorm: Output = RMSNorm(O) * Sigmoid(Gate)
        o = self.o_norm(o, gate_output)
        # o (after norm): [batch_size, seq_len, num_heads, head_v_dim]

        # 展平头维度：[batch, seq, num_heads, head_dim] -> [batch, seq, hidden_size]
        o = rearrange(o, "b t h d -> b t (h d)")
        # 最终输出投影
        o = self.o_proj(o)
        # o (final): [batch_size, seq_len, hidden_size]

        # 如果之前移除了 padding，现在恢复 padding 结构
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
