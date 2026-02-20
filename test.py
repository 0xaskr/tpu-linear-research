import os
import sys
import functools
import numpy as np
# Add the fla directory to the path so we can import from the inner fla package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'fla')))

#设置triton在cpu上跑
os.environ["TRITON_CPU_BACKEND"] = "1"
os.environ["TRITON_INTERPRET"] = "1"
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import torch
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk as fla_chunk_gla_fwd_o_gk
from fla.layers.kda import KimiDeltaAttention as fla_kda

def cdiv(a: jax.Array, b: jax.Array | int):
  return jnp.ceil(a / b)

def cdiv_pt(a, b):
  return (a + b - 1) // b

# cu_seqlens 这个最好是cpu tensor
def prepare_chunk_indices(
    cu_seqlens: jax.Array,
    chunk_size: int,
) -> torch.LongTensor:
    """
    Generate chunk indices for variable length sequences.
    为变长序列生成分块索引。

    Returns:
        torch.LongTensor: A tensor of shape [Num_Total_Chunks, 2].
        Each row is (sequence_id, chunk_id).
        每一行包含两个标记：(句子ID, 该句内的块ID)。
    """

    # prepare_lens = torch.diff
    indices = jnp.concatenate([jnp.arange(n) for n in cdiv(jnp.diff(cu_seqlens), chunk_size).tolist()])
    # 这个函数只生成逻辑上的 (Seq_ID, Chunk_ID) 对，不涉及实际数据的读取。
    # 越界保护机制 (Boundary Check):
    # 下游的 Triton Kernel 在使用这些 ID 时，必须执行以下逻辑：
    # 1. 计算当前 Chunk 的起始 Token： start_token_idx = cu_seqlens[seq_id] + chunk_id * chunk_size
    # 2. 计算当前 Chunk 的有效长度： visible_len = min(chunk_size, seq_len - chunk_id * chunk_size)
    # 3. 使用 Mask 加载数据： tl.load(..., mask=offsets < visible_len, other=0)
    # 因此，即使最后一个 Chunk 不满 (Partial Chunk)，Kernel 也会通过 Mask 防止越界。
    return jnp.array(jnp.stack([(indices == 0).cumsum(0) - 1, indices], 1), dtype = cu_seqlens.dtype, device=cu_seqlens.device)

def chunk_gla_fwd_kernel_o(
    # in
    q,
    v,
    g,
    h,
    A,
    cu_seqlens,
    chunk_indices,
    # out
    o,
    # static args
    T,
    H,
    K,
    V,
    BT,
    BK,
    BV,
    USE_EXP2,
    IS_VARLEN,
    scale
):
  B, T, K = q.shape[1:]
  V = v.shape[-1]
  NT = h.shape[2]
  TOTAL_T = B * T
  TOTAL_NT = B * NT

  q = q.reshape(TOTAL_T, K)
  v = v.reshape(TOTAL_T, V)
  g = g.reshape(TOTAL_T, K)
  h = h.reshape(TOTAL_NT, K, V)
  A = A.reshape(TOTAL_T, BT)
  o = o.reshape(TOTAL_T, V)

  print("q.shape = ", q.shape)

  # i_t = 当前chunk/当前batch的哪个chunk
  i_v, i_t, i_bh = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  i_b, i_h = i_bh // H, i_bh % H
  real_T = 0
  real_NT = 0
  if IS_VARLEN:
    i_tg = i_t
    i_n = chunk_indices[i_t, 0].astype(jnp.int32)
    i_t = chunk_indices[i_t, 1].astype(jnp.int32)
    bos = cu_seqlens[i_n].astype(jnp.int32)
    eos = cu_seqlens[i_n + 1].astype(jnp.int32)
    real_T = eos - bos
    real_NT = cdiv_pt(real_T, BT)
  else:
    real_NT = cdiv_pt(T, BT)
    i_tg = i_b * real_NT + i_t
    bos,eos = i_b * T, (i_b + 1) * T

  m_s = jnp.arange(0, BT)[:, None] >= jnp.arange(0, BT)[None,:]
  b_o = jnp.zeros([BT, BV], dtype=jnp.float32)

  for i_k in range(cdiv_pt(K, BK)):
    b_q = q[pl.ds(bos+i_t * BT, BT), pl.ds(i_k * BK, BK)]
    b_g = g[pl.ds(bos+i_t * BT, BT), pl.ds(i_k * BK, BK)].astype(jnp.float32)
    b_h = h[i_tg, pl.ds(i_k * BK, BK), pl.ds(i_v * BV, BV)]
    if (USE_EXP2):
      b_qg = (b_q * jnp.exp2(b_g)).astype(b_q.dtype)
    else:
      b_qg = (b_q * jnp.exp(b_g)).astype(b_q.dtype)
    if i_k >= 0:
      b_o += jnp.dot(b_qg, b_h.astype(b_qg.dtype))

  b_o *= scale
  b_v = v[pl.ds(bos+i_t* BT, BT), pl.ds(i_v * BV, BV)]
  b_A = A[pl.ds(bos+i_t* BT, BT), 0:BT]
  # Apply causal mask
  b_A = jnp.where(m_s, b_A, 0.0).astype(b_A.dtype)
  b_o += jnp.dot(b_A, b_v)
  o[pl.ds(bos+i_t* BT, BT), pl.ds(i_v * BV, BV)] = b_o.astype(o.dtype)


def chunk_gla_fwd_o_gk(
    q: jax.Array,          # [B, T, H, K]
    v: jax.Array,          # [B, T, H, V]
    g: jax.Array,          # [B, T, H, K]
    A: jax.Array,          # ([B, T, H, BT] or similar structure depending on chunking)
    h: jax.Array,          # [B, H, K, V]
    scale: float,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
    chunk_indices: jax.Array | None = None,
    use_exp2: bool = False,
) -> jax.Array:

  B, T, H, K = q.shape
  V = v.shape[-1]
  BT = chunk_size
  BK = 64
  BV = 128
  N = B if cu_seqlens is None else cu_seqlens.shape[-1] - 1

  assert q.shape == (B, T, H, K)
  assert v.shape == (B, T, H, V)
  assert g.shape == (B, T, H, K)
  assert A.shape == (B, T, H, BT)
  assert (cu_seqlens is None) or (cu_seqlens.shape == (N+1,))
  # assert (chunk_indices is None) or (chunk_indices.shape == (cdiv_pt(T,BT),2))
  assert BK in [32, 64]
  assert BV in [64, 128]
  assert (cu_seqlens is None) or ((cu_seqlens is not None) and (chunk_indices is not None))
  assert (cu_seqlens is None) or ((cu_seqlens is not None) and B == 1)
  assert (cu_seqlens is None) or (cu_seqlens.dtype == jnp.int32)
  assert (chunk_indices is None) or (chunk_indices.dtype == jnp.int32)

  IS_VARLEN = cu_seqlens is not None

  # NT = 所有batch加在一起，chunk的数量
  NT = cdiv_pt(T, BT) if cu_seqlens is None else len(chunk_indices)

  # Padding Logic for Varlen to prevent OOB access in Pallas Kernel
  # The kernel reads fixed blocks of size BT. If the last chunk of a sequence
  # starts near the end of T (e.g., index 224 for T=256 with BT=64),
  # it will try to read up to 288. We must pad q, v, g, A.
  # We pad by BT to be safe.
  # Only apply padding if IS_VARLEN is true, as standard mode assumes perfect tiling or handles it differently.

  orig_T = T
  pad_len = 0
  if IS_VARLEN:
      pad_len = BT
  else:
      if T % BT != 0:
          pad_len = cdiv_pt(T, BT) * BT - T

  if pad_len > 0:
      # Pad last dimension of T (axis 1)
      # q: (B, T, H, K)
      q = jnp.pad(q, ((0,0), (0, pad_len), (0,0), (0,0)))
      v = jnp.pad(v, ((0,0), (0, pad_len), (0,0), (0,0)))
      g = jnp.pad(g, ((0,0), (0, pad_len), (0,0), (0,0)))
      # A: (B, T, H, BT)
      A = jnp.pad(A, ((0,0), (0, pad_len), (0,0), (0,0)))
      # Update T to reflect padded size for reshape inside kernel
      T = T + pad_len


  assert h.shape == (B, NT, H, K, V)

  v_t_shape = [H, B, T, V]

  q_block_t_shape = [1, B, T, K]
  v_block_t_shape = [1, B, T, V]
  g_block_t_shape = [1, B, T, K]
  h_block_t_shape = [1, B, NT, K, V]
  A_block_t_shape = [1, B, T, BT]

  q_t = q.transpose(2, 0, 1, 3)
  v_t = v.transpose(2, 0, 1, 3)
  g_t = g.transpose(2, 0, 1, 3)
  h_t = h.transpose(2, 0, 1, 3, 4)
  A_t = A.transpose(2, 0, 1, 3)

  # o = jnp.zeros(v.shape, v.dtype)
  o_shape=jax.ShapeDtypeStruct(v_t.shape, v.dtype)
  o_blockspec = pl.BlockSpec(v_block_t_shape, index_map = lambda v, nt, bh: (bh%H, 0, 0, 0))

  q_blockspec = pl.BlockSpec(q_block_t_shape, index_map=lambda v, nt, bh: (bh%H, 0, 0, 0))
  v_blockspec = pl.BlockSpec(v_block_t_shape, index_map=lambda v, nt, bh: (bh%H, 0, 0, 0))
  g_blockspec = pl.BlockSpec(g_block_t_shape, index_map=lambda v, nt, bh: (bh%H, 0, 0, 0))
  h_blockspec = pl.BlockSpec(h_block_t_shape, index_map = lambda v, nt, bh: (bh%H, 0, 0, 0, 0))
  A_blockspec = pl.BlockSpec(A_block_t_shape, index_map=lambda v, nt, bh: (bh%H, 0, 0, 0))
  cs_blockspec = pl.BlockSpec([N+1], index_map=lambda v, nt, bh: (0,), memory_space=pltpu.MemorySpace.SMEM)
  ci_blockspec = pl.BlockSpec([NT, 2], index_map=lambda v, nt, bh: (0, 0,), memory_space=pltpu.MemorySpace.SMEM)

  if cu_seqlens is None:
      cu_seqlens_arg = jnp.zeros((N+1,), dtype=jnp.int32)
      chunk_indices_arg = jnp.zeros((NT, 2), dtype=jnp.int32)
  else:
      cu_seqlens_arg = cu_seqlens
      chunk_indices_arg = chunk_indices

  grid = (cdiv_pt(V, BV), NT, B * H)
  o = pl.pallas_call(
    functools.partial(
        chunk_gla_fwd_kernel_o,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_EXP2=use_exp2,
        IS_VARLEN=(cu_seqlens is not None),
        scale=scale # opt this
    ),
    grid=grid,
    out_shape=o_shape,
    out_specs=o_blockspec,
    in_specs=[q_blockspec, v_blockspec, g_blockspec, h_blockspec, A_blockspec,cs_blockspec, ci_blockspec],
  )(q_t,v_t,g_t,h_t,A_t,cu_seqlens_arg,chunk_indices_arg)


  o = o.transpose(1, 2, 0, 3)

  if pad_len > 0:
      # Slice back to original size
      o = o[:, :orig_T, ...]

  return o

def test_fla_kda_shape():
  B, T, hidden_size = 2, 128, 32
  rng_dtype = torch.float32
  device = "cpu" if os.environ.get("TRITON_CPU_BACKEND", "0") == "1" else "cuda"
  hidden_states = torch.randn(B, T, hidden_size, device=device, dtype=rng_dtype)

  # mode='chunk'
  # print("Testing chunk mode:")
  # kda = fla_kda(hidden_size=hidden_size, mode='chunk', head_dim=4, num_heads=4).to(rng_dtype)
  # o, _, _ = kda(hidden_states)
  # print("Output shape:", o.shape)

  mode='chunk'
  print(f"\nTesting {mode} mode:")
  kda_recurrent = fla_kda(hidden_size=hidden_size, mode=mode, head_dim=4, num_heads=4).to(device=device, dtype=rng_dtype)
  # q_len <= 64 will force fused_recurrent in inference, but here we explicitly set mode.
  # Note: The code overrides mode to 'fused_recurrent' if q_len <= 64 and not training.
  # To test fused_recurrent explicitly, we can use a short sequence or rely on the logic.
  # Let's ensure it runs.
  kda_recurrent.eval() # Set to eval to avoid assertion error
  # Actually the logic says: mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
  # So if we are in train mode, it respects self.mode.
  o_recurrent, _, _ = kda_recurrent(hidden_states)
  print("Output shape:", o_recurrent.shape)

def run_test_chunk_gla_fwd_o_gk(B, T, H, K, V, chunk_size, cu_seqlens_list, use_exp2):
  print(f"\n=============================================")
  print(f"Testing chunk_gla_fwd_o_gk")
  print(f"B={B}, T={T}, H={H}, K={K}, V={V}, chunk_size={chunk_size}")
  print(f"cu_seqlens={cu_seqlens_list}, use_exp2={use_exp2}")
  print(f"=============================================")

  if cu_seqlens_list is not None:
    cu_seqlens = np.array(cu_seqlens_list, dtype=np.int32)
    cu_seqlens_jax = jnp.array(cu_seqlens)
    chunk_indices_jax = prepare_chunk_indices(cu_seqlens_jax, chunk_size)
    NT = len(chunk_indices_jax)
    cu_seqlens_pt = torch.LongTensor(cu_seqlens)
  else:
    cu_seqlens_jax = None
    chunk_indices_jax = None
    NT = cdiv_pt(T, chunk_size)
    cu_seqlens_pt = None

  rng_dtype = torch.float32
  jax_dtype = jnp.float32
  device = "cpu" if os.environ.get("TRITON_CPU_BACKEND", "0") == "1" else "cuda"

  q = torch.randn(B, T, H, K, device=device, dtype=rng_dtype)
  v = torch.randn(B, T, H, V, device=device, dtype=rng_dtype)
  g_raw = torch.randn(B, T, H, K, device=device, dtype=rng_dtype) * 0.1
  g = g_raw.cumsum(dim=1)

  A = torch.randn(B, T, H, chunk_size, device=device, dtype=rng_dtype)
  h = torch.randn(B, NT, H, K, V, device=device, dtype=rng_dtype) # h: inter-chunk hidden state [B, NT, H, K, V]

  scale = K ** -0.5

  q_jax = jnp.array(q.to(torch.float32), dtype = jax_dtype)
  v_jax = jnp.array(v.to(torch.float32), dtype = jax_dtype)
  g_jax = jnp.array(g.to(torch.float32), dtype = jax_dtype)
  h_jax = jnp.array(h.to(torch.float32), dtype = jax_dtype)
  A_jax = jnp.array(A.to(torch.float32), dtype = jax_dtype)

  o_jax = chunk_gla_fwd_o_gk(q_jax, v_jax, g_jax, A_jax, h_jax, scale,
        cu_seqlens=cu_seqlens_jax, chunk_size=chunk_size, chunk_indices=chunk_indices_jax, use_exp2=use_exp2)

  # 调用 triton kernel
  pt_o = fla_chunk_gla_fwd_o_gk(
      q=q,
      v=v,
      g=g,
      A=A,
      h=h,
      scale=scale,
      cu_seqlens=cu_seqlens_pt,
      chunk_size=chunk_size,
      use_exp2=use_exp2
  )

  o_jax_np = np.array(o_jax)
  pt_o_np = pt_o.detach().cpu().numpy()

  diff = np.abs(o_jax_np - pt_o_np)
  max_diff = diff.max()
  mean_diff = diff.mean()

  print(f"Max Difference: {max_diff}")
  print(f"Mean Difference: {mean_diff}")

  if max_diff > 1e-4:
      print("MISMATCH DETECTED!")
      max_idx = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
      print(f"Max mismatch at index {max_idx}")
      print(f"JAX value: {o_jax_np[max_idx]}")
      print(f"Torch value: {pt_o_np[max_idx]}")
      return False
  else:
      print("Results MATCH.")
      return True

def test_chunk_gla_fwd_o_gk_all():
  test_cases = [
      # Original case
      dict(B=1, T=256, H=4, K=32, V=32, chunk_size=64, cu_seqlens_list=[0, 32, 256], use_exp2=True),
      # Non-varlen mode (fixed length)
      dict(B=2, T=256, H=4, K=32, V=32, chunk_size=64, cu_seqlens_list=None, use_exp2=True),
      dict(B=2, T=128, H=4, K=64, V=64, chunk_size=32, cu_seqlens_list=None, use_exp2=False),
      dict(B=1, T=64, H=2, K=64, V=128, chunk_size=64, cu_seqlens_list=None, use_exp2=True),
      # Partial chunks standard mode
      dict(B=2, T=100, H=2, K=32, V=64, chunk_size=32, cu_seqlens_list=None, use_exp2=False),
      # Partial chunk varlen mode
      dict(B=1, T=100, H=2, K=64, V=64, chunk_size=32, cu_seqlens_list=[0, 40, 100], use_exp2=True),
      # Another varlen
      dict(B=1, T=35, H=2, K=32, V=128, chunk_size=64, cu_seqlens_list=[0, 35], use_exp2=False),
  ]
  all_passed = True
  for kwargs in test_cases:
      try:
          passed = run_test_chunk_gla_fwd_o_gk(**kwargs)
      except Exception as e:
          print(f"Exception during test: {e}")
          passed = False
      all_passed = all_passed and passed

  if all_passed:
      print("\nALL TESTS PASSED!")
  else:
      print("\nSOME TESTS FAILED!")

if __name__ == "__main__":
  # test_fla_kda_shape()
  test_chunk_gla_fwd_o_gk_all()

