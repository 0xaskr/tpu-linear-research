# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Code is adapted from flash-attn.bert_padding.py


import torch
from einops import rearrange, repeat

from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
from fla.utils import tensor_cache


class IndexFirstAxis(torch.autograd.Function):
    """
    Custom autograd function to index the first axis of a tensor.
    自定义自动求导函数，用于对张量的第一个维度进行索引。
    
    This is often used to select valid tokens from a flattened batch of sequences based on indices.
    通常用于根据索引从展平的批次序列中选择有效 token。
    """

    @staticmethod
    def forward(ctx, x, indices):
        """
        Args:
            x: Input tensor, typically shape [Batch * SeqLen, ...]. 输入张量，通常展平了批次和序列维度。
            indices: Indices of elements to select. 选择元素的索引。
        """
        ctx.save_for_backward(indices)
        assert x.ndim >= 2
        ctx.first_axis_dim, other_shape = x.shape[0], x.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return x[indices]
        # 使用 scatter/gather 替代直接索引，通常在某些硬件上更快
        return torch.gather(
            rearrange(x, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, do):
        """
        Backward pass: Scatters gradients back to their original positions.
        反向传播：将梯度 scatter 回原始位置。
        """
        (indices,) = ctx.saved_tensors
        assert do.ndim >= 2
        other_shape = do.shape[1:]
        do = rearrange(do, "b ... -> b (...)")
        dx = torch.zeros(
            [ctx.first_axis_dim, do.shape[1]],
            device=do.device,
            dtype=do.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # dx[indices] = do
        dx.scatter_(0, repeat(indices, "z -> z d", d=do.shape[1]), do)
        return dx.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    """
    Custom autograd function to put elements into the first axis of a tensor at specified indices.
    自定义自动求导函数，将元素放置到张量第一个维度的指定索引处。
    
    This is the inverse operation of IndexFirstAxis, effectively "padding" a packed tensor back to a full tensor.
    這是 IndexFirstAxis 的逆操作，实际上是将紧凑张量 (packed tensor) "填充" 回完整张量。
    """

    @staticmethod
    def forward(ctx, x, indices, first_axis_dim):
        """
        Args:
            x: Input ragged/packed tensor. 输入的紧凑张量。
            indices: Indices where elements should be placed. 放置元素的索引。
            first_axis_dim: The size of the first dimension of the output tensor. 输出张量第一个维度的大小。
        """
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert x.ndim >= 2
        y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
        # TODO [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        y[indices] = x
        # y.scatter_(0, repeat(indices, 'z -> z d', d=x.shape[1]), x)
        return y

    @staticmethod
    def backward(ctx, do):
        """
        Backward pass: Gathers gradients from the specified indices.
        反向传播：从指定索引处 gather 梯度。
        """
        (indices,) = ctx.saved_tensors
        # TODO [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        dx = do[indices]
        # dx = torch.gather(do, 0, repeat(indices, 'z -> z d', d=do.shape[1]))
        return dx, None, None


index_put_first_axis = IndexPutFirstAxis.apply


@tensor_cache
def get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.
    获取将未填充 (ragged) 张量重新填充 (repad) 所需的索引数据。

    This function processes the attention mask to find valid tokens and their positions,
    which is essential for FlashAttention-style packed processing.
    该函数处理 attention mask 以查找有效 token 及其位置，这对于 FlashAttention 风格的紧凑处理至关重要。

    Args:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
            形状为 (batch_size, sequence_length) 的布尔或整数张量，1 表示有效，0 表示无效。

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
            展平后输入序列中非掩码 (有效) token 的索引。
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors.
            `cu_seqlens` shape is [batch_size + 1].
            累积序列长度，用于索引到 ragged (未填充) 张量。
            `cu_seqlens` 的形状为 [batch_size + 1]。
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
            批次中的最大序列长度。
    """
    lens = prepare_lens_from_mask(attention_mask)
    # lens = [batch]
    #  return mask.sum(dim=-1, dtype=torch.int32)
    # 这一句是 Unpadding (去填充) 的核心步骤：
    # 1. .flatten(): 将 [Batch, Seq] 的二维 Mask 拍扁成一维 [Batch*Seq]。
    # 2. .nonzero(as_tuple=False): 找出所有非 0 的位置。
    #    - as_tuple=False: 返回一个二维张量 [N, 1] (N是有效元素个数)。
    #      如果设为 True，会返回一个元组 (tensor([idx1, idx2...]),)，这里我们需要直接操作 Tensor。
    #    - 最后再 .flatten() 把这个 [N, 1] 变成 [N] 的一维索引数组。
    # 3. 结果 indices: 这是一个只包含有效 Token 坐标的一维数组。
    #    后续会用这个 indices 去 hidden_states 里把真实数据“抓”出来，丢掉 padding。
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    print("lens shape = ", lens.shape)
    max_seqlen_in_batch = lens.max().item()
    cu_seqlens = prepare_cu_seqlens_from_mask(attention_mask)
    # cumsum + pad
    return indices, cu_seqlens, max_seqlen_in_batch


def unpad_input(
    q: torch.Tensor,
    states: tuple[torch.Tensor],
    attention_mask: torch.Tensor,
    q_len: int,
    keepdim: bool = False,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens
    even though they belong to different batches.
    去除 query, key 和 value 张量的填充，使用单个维度存储所有 token，即使它们属于不同的批次。
    
    This converts [B, S, D] tensors into [Total_Valid_Tokens, D] ragged tensors.
    这将 [B, S, D] 张量转换为 [总有效Token数, D] 的紧凑张量。


    Arguments:
        q (`torch.Tensor`):
            Query state with padding. Shape: [batch_size, q_len, ...].
            带填充的 Query 状态。形状: [batch_size, q_len, ...]。
        states (`Tuple[torch.Tensor]`):
            Attention state with padding. Shape: [batch_size, seq_len, ...].
            带填充的 Attention 状态 (通常是 Key/Value)。形状: [batch_size, seq_len, ...]。
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape [batch_size, sequence_length], 1 means valid and 0 means not valid.
            形状为 [batch_size, sequence_length] 的掩码，1 有效，0 无效。
        q_len (`int`):
            Target length. 目标长度 (通常是 Query 的长度)。
        keepdim (`bool`):
            Whether to keep the batch dimension. Default: `False`.
            是否保留批次维度。默认: `False`. 如果为 True，通常会 unsqueeze 一个维度。

    Return:
        q (`torch.Tensor`):
            Query state without padding.
            Shape: [1, total_target_length, ...] if `keepdim=True` else [total_target_length, ...].
            去填充后的 Query。
        states (`Tuple[torch.Tensor]`):
            Attention state without padding.
            Shape: [1, total_source_length, ...] if `keepdim=True` else [total_source_length, ...].
            去填充后的 Attention 状态。
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
            展平后输入目标序列中非掩码 token 的索引。
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value),
            used to index into ragged (unpadded) tensors.
            `cu_seqlens` shape is [batch_size + 1].
            Target (query) 和 Source (key, value) 的累积序列长度。
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence
            i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
            批次中的最大序列长度。
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = get_unpad_data(attention_mask)
    batch_size, seq_len, *_ = states[0].shape

    state = tuple(
        index_first_axis(rearrange(s, "b s ... -> (b s) ..."), indices_k)
        for s in states
    )

    if q_len == seq_len:
        q = index_first_axis(rearrange(q, "b s ... -> (b s) ..."), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif q_len == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
        indices_q = cu_seqlens_q[:-1]
        q = q.squeeze(1)
    else:
        raise NotImplementedError("We only support either q_len == k_len (prefilling) or q_len == 1 (decoding)")

    if keepdim:
        q = q.unsqueeze(0)
        state = tuple(s.unsqueeze(0) for s in state)

    return (
        q,
        state,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Pads a ragged tensor back to a dense tensor with padding.
    将紧凑 (ragged) 张量重新填充为带 padding 的稠密张量。

    Args:
        hidden_states ([total_tokens, ...]):
            where total_tokens denotes the number of tokens in selected in attention_mask.
            输入张量，形状为 [总有效Token数, ...]，其中 total_tokens 是 attention_mask 中选中的 token 数。
        indices ([total_tokens]):
            the indices that represent the non-masked tokens of the original padded input sequence.
            表示原始填充输入序列中非掩码 (有效) token 的索引。
        batch_size (int):
            batch_size size for the padded sequence.
            填充后序列的 batch size。
        seq_len (int):
            maximum sequence length for the padded sequence.
            填充后序列的最大序列长度。

    Return:
        hidden_states of shape [batch_size, seq_len, ...]
        形状为 [batch_size, seq_len, ...] 的 hidden_states
    """
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)
