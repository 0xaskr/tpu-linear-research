import os
import argparse
import time
import sys
import torch
import numpy as np
import jax
import jax.numpy as jnp

# Set Triton environment variables for CPU execution (Correctness only)
os.environ["TRITON_CPU_BACKEND"] = "1"
os.environ["TRITON_INTERPRET"] = "1"

# Add local directory to path
sys.path.append(os.getcwd())
if os.path.exists(os.path.join(os.getcwd(), 'fla')):
    sys.path.append(os.path.join(os.getcwd(), 'fla'))

# Attempt imports
try:
    from test_pallas_manual_varlen import chunk_gated_delta_rule_fwd_h_varlen as pallas_fwd
    from test_pallas_manual_varlen import chunk_gla_fwd_o_gk_varlen as pallas_o_gk_fwd
    from test_pallas_manual_varlen import prepare_chunk_indices
    print("Successfully imported Pallas kernel wrapper.")
except ImportError as e:
    print(f"Failed to import Pallas kernel wrapper: {e}")
    pallas_fwd = None
    pallas_o_gk_fwd = None
    prepare_chunk_indices = None

try:
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h as triton_fwd
    from fla.ops.gla.chunk import chunk_gla_fwd_o_gk as triton_o_gk_fwd
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton_fwd = None
    triton_o_gk_fwd = None
    print("Triton kernel not found.")

def compare_tensor(name, pt_t, jax_t, atol=1e-5, rtol=1e-5):
    if pt_t is None and jax_t is None:
        print(f"[{name}] Both are None. MATCH.")
        return
    if pt_t is None or jax_t is None:
        print(f"[{name}] One is None! MISMATCH.")
        return

    pt_val = pt_t.detach().cpu().float().numpy() if isinstance(pt_t, torch.Tensor) else np.array(pt_t)
    jax_val = np.array(jax_t.astype(jnp.float32)) if hasattr(jax_t, 'dtype') and jax_t.dtype == jnp.bfloat16 else np.array(jax_t)

    if pt_val.shape != jax_val.shape:
        print(f"[{name}] Shape mismatch: Left {pt_val.shape} vs Right {jax_val.shape}. FAIL.")
        if pt_val.squeeze().shape == jax_val.squeeze().shape:
            print(f"  Attempting comparison with squeezed shapes: {pt_val.squeeze().shape}")
            pt_val = pt_val.squeeze()
            jax_val = jax_val.squeeze()
        else:
            return

    diff = np.abs(pt_val - jax_val)
    max_diff = np.max(diff)
    max_val = np.max(np.abs(jax_val))
    max_rel_diff = np.max(diff / (np.abs(jax_val) + 1e-12))

    is_close = np.allclose(pt_val, jax_val, atol=atol, rtol=rtol)
    status = "PASS" if is_close else "FAIL"

    print(f"[{name}] {status}")
    print(f"  Max Value        : {max_val:.6e}")
    print(f"  Max Abs Diff     : {max_diff:.6e}")
    print(f"  Max Rel Diff     : {max_rel_diff:.6e}")

    if not is_close:
        tolerance = atol + rtol * np.abs(jax_val)
        error_ratio = diff / (tolerance + 1e-12)
        idx = np.unravel_index(np.argmax(error_ratio), error_ratio.shape)
        print(f"  Max Mismatch details at index {idx}:")
        print(f"    Left (Triton)  = {pt_val[idx]}")
        print(f"    Right (Pallas) = {jax_val[idx]}")
        print(f"    Diff           = {diff[idx]}")
        print(f"    Tolerance      = {tolerance[idx]} (atol={atol} + rtol={rtol}*|Right|)")
        print(f"    Ratio          = {error_ratio[idx]}")

def generate_inputs(B, H, K, V, seqlens_list, chunk_size, dtype=torch.bfloat16):
    N = len(seqlens_list)
    TotalT = sum(seqlens_list)

    torch.manual_seed(42)
    k = torch.randn((B, TotalT, H, K), dtype=dtype)
    w = torch.randn((B, TotalT, H, K), dtype=dtype)
    u = torch.randn((B, TotalT, H, V), dtype=dtype)
    h0 = torch.randn((N, H, K, V), dtype=dtype)

    # Generate chunk-local cumulative log decay for g and gk
    def chunk_local_cumsum(x, chunk_size):
        out = x.clone()
        for i in range(0, x.shape[1], chunk_size):
            end = min(i + chunk_size, x.shape[1])
            out[:, i:end] = x[:, i:end].cumsum(1)
        return out

    raw_g = -torch.randn((B, TotalT, H), dtype=dtype).abs() * 0.01
    g = chunk_local_cumsum(raw_g, chunk_size)

    raw_gk = -torch.randn((B, TotalT, H, K), dtype=dtype).abs() * 0.01
    gk = chunk_local_cumsum(raw_gk, chunk_size)

    cu_seqlens = torch.tensor([0] + list(np.cumsum(seqlens_list)), dtype=torch.int32)

    return k, w, u, g, gk, h0, cu_seqlens

def run_correctness(args):
    print("\n" + "="*40)
    print("Running Varlen Comparison FP32 (chunk_gated_delta_rule_fwd_h)")
    print("="*40)

    # Configuration
    rng_dtype = torch.bfloat16
    triton_dtype = torch.float32
    jax_dtype = jnp.bfloat16

    # seqlens_list = [64, 128, 64]
    # Using the case that failed previously to ensure fix
    seqlens_list = [32, 128, 137]
    N = len(seqlens_list)
    TotalT = sum(seqlens_list)
    chunk_size = 64
    B, H, K, V = 1, 4, 128, 64

    print(f"N={N}, Seqlens={seqlens_list}, TotalT={TotalT}, H={H}, K={K}, V={V}, chunk_size={chunk_size}")

    k, w, u, g, gk, h0, cu_seqlens = generate_inputs(B, H, K, V, seqlens_list, chunk_size, rng_dtype)

    # Convert for Triton
    k_pt = k.to(device="cpu", dtype=triton_dtype)
    w_pt = w.to(device="cpu", dtype=triton_dtype)
    u_pt = u.to(device="cpu", dtype=triton_dtype)
    g_pt = g.to(device="cpu", dtype=triton_dtype)
    gk_pt = gk.to(device="cpu", dtype=triton_dtype)
    h0_pt = h0.to(device="cpu", dtype=triton_dtype)

    # Triton Run
    if HAS_TRITON and triton_fwd is not None:
        print("\nRunning Triton varlen (FP32)...")
        h_ref, v_new_ref, final_state_ref = triton_fwd(
            k=k_pt, w=w_pt, u=u_pt, g=g_pt, gk=gk_pt,
            initial_state=h0_pt, output_final_state=True,
            chunk_size=chunk_size, save_new_value=True,
            cu_seqlens=cu_seqlens.long(),
            use_exp2=False
        )
        print(f"Triton h_ref shape: {h_ref.shape}")
    else:
        print("Skipping Triton run (not available).")
        h_ref, v_new_ref, final_state_ref = None, None, None

    # Pallas Run
    if pallas_fwd is not None:
        print("\nRunning Pallas varlen (FP32)...")
        k_jax = jnp.array(k.to(torch.float32), dtype=jax_dtype)
        w_jax = jnp.array(w.to(torch.float32), dtype=jax_dtype)
        u_jax = jnp.array(u.to(torch.float32), dtype=jax_dtype)
        g_jax = jnp.array(g.to(torch.float32), dtype=jax_dtype)
        gk_jax = jnp.array(gk.to(torch.float32), dtype=jax_dtype)
        h0_jax = jnp.array(h0.to(torch.float32), dtype=jax_dtype)
        cu_seqlens_jax = jnp.array(cu_seqlens.numpy(), dtype=jnp.int32)

        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        chunk_indices_jax = jnp.array(chunk_indices.numpy(), dtype=jnp.int32)

        h_history_jax, v_new_jax, final_state_jax = pallas_fwd(
            k=k_jax, w=w_jax, v=u_jax, g=g_jax, gk=gk_jax,
            initial_state=h0_jax, output_final_state=True,
            chunk_size=chunk_size, save_new_value=True,
            seqlens=cu_seqlens_jax,
            chunk_indices=chunk_indices_jax,
            use_exp2=False
        )
        jax.block_until_ready(h_history_jax)
        print(f"Pallas h_jax shape: {h_history_jax.shape}")
    else:
        print("Skipping Pallas run (not available).")
        h_history_jax, v_new_jax, final_state_jax = None, None, None

    print("\n" + "="*40)
    print("COMPARISON RESULTS (FP32)")
    print("="*40)

    # Tolerances for FP32
    atol, rtol = 1e-2, 1e-2

    if h_ref is not None and h_history_jax is not None:
        compare_tensor("Hidden States (h)", h_ref, h_history_jax, atol=atol, rtol=rtol)
        compare_tensor("Residual (v_new)", v_new_ref, v_new_jax, atol=atol, rtol=rtol)
        compare_tensor("Final State (ht)", final_state_ref, final_state_jax, atol=atol, rtol=rtol)
    else:
        print("Skipping comparison because one or both backends failed to run.")

def run_comparison_o_gk_varlen(args):
    print("\n" + "="*40)
    print("Running Varlen Comparison o_gk (chunk_gla_fwd_o_gk)")
    print("="*40)

    if pallas_o_gk_fwd is None:
        print("Pallas o_gk kernel not found.")
        return
    if triton_o_gk_fwd is None:
        print("Triton o_gk kernel not found.")
        return

    # Configuration
    rng_dtype = torch.bfloat16
    triton_dtype = torch.float32
    jax_dtype = jnp.float32

    seqlens_list = [64, 128, 128]
    N = len(seqlens_list)
    TotalT = sum(seqlens_list)
    chunk_size = 64
    B, H, K, V = 1, 4, 64, 64
    scale = K ** -0.5

    print(f"N={N}, Seqlens={seqlens_list}, TotalT={TotalT}, H={H}, K={K}, V={V}, chunk_size={chunk_size}")

    torch.manual_seed(42)
    # Inputs: q, v, g, A, h
    q = torch.randn((B, TotalT, H, K), dtype=rng_dtype)
    v = torch.randn((B, TotalT, H, V), dtype=rng_dtype)
    g = torch.randn((B, TotalT, H, K), dtype=rng_dtype)
    A = torch.randn((B, TotalT, H, chunk_size), dtype=rng_dtype)

    # Generate chunk indices/offsets
    cu_seqlens = torch.tensor([0] + list(np.cumsum(seqlens_list)), dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    TotalChunks = chunk_indices.shape[0]

    h = torch.randn((B, TotalChunks, H, K, V), dtype=rng_dtype)

    # Triton Run
    print("Running Triton o_gk varlen...")
    q_pt = q.to(device="cpu", dtype=triton_dtype)
    v_pt = v.to(device="cpu", dtype=triton_dtype)
    g_pt = g.to(device="cpu", dtype=triton_dtype)
    A_pt = A.to(device="cpu", dtype=triton_dtype)
    h_pt = h.to(device="cpu", dtype=triton_dtype)
    cu_seqlens_pt = cu_seqlens.to(device="cpu", dtype=torch.int32)

    # Triton chunk_gla_fwd_o_gk signature:
    # q, v, g, A, h, scale, cu_seqlens, chunk_size, chunk_indices, use_exp2
    o_ref = triton_o_gk_fwd(
        q=q_pt, v=v_pt, g=g_pt, A=A_pt, h=h_pt,
        scale=scale,
        cu_seqlens=cu_seqlens_pt,
        chunk_size=chunk_size,
        use_exp2=False
    )

    # Pallas Run
    print("Running Pallas o_gk varlen...")
    # Pallas inputs: [TotalT, H, K] (squeezed B=1)
    q_jax = jnp.array(q.squeeze(0).float().numpy(), dtype=jax_dtype)
    v_jax = jnp.array(v.squeeze(0).float().numpy(), dtype=jax_dtype)
    g_jax = jnp.array(g.squeeze(0).float().numpy(), dtype=jax_dtype)
    A_jax = jnp.array(A.squeeze(0).float().numpy(), dtype=jax_dtype)
    h_jax = jnp.array(h.squeeze(0).float().numpy(), dtype=jax_dtype)

    chunk_indices_jax = jnp.array(chunk_indices.numpy(), dtype=jnp.int32)
    seqlens_jax = jnp.array(cu_seqlens.numpy(), dtype=jnp.int32)

    o_jax = pallas_o_gk_fwd(
        q=q_jax, v=v_jax, g=g_jax, A=A_jax, h=h_jax,
        chunk_indices=chunk_indices_jax,
        seqlens=seqlens_jax,
        scale=scale,
        chunk_size=chunk_size,
        use_exp2=False
    )

    # Compare
    # o_ref: [B, TotalT, H, V] -> [TotalT, H, V]
    o_ref_squeezed = o_ref.squeeze(0)
    compare_tensor("Output (o)", o_ref_squeezed, o_jax, atol=1e-2, rtol=1e-2)

def run_benchmark(args):
    if pallas_fwd is None:
        print("Pallas kernel not available for benchmarking.")
        return

    print("\n" + "="*40)
    print("Running Varlen Benchmark (Pallas)")
    print("="*40)

    # Configuration
    seqlens_list = [args.seq_len] * args.batch_size
    chunk_size = 64
    B, H, K, V = 1, args.num_heads, args.head_dim, args.head_dim
    # B is 1 because packed sequence
    TotalT = sum(seqlens_list)

    print(f"Config: Seqlens=[{args.seq_len}]*{args.batch_size}, TotalT={TotalT}, H={H}, K={K}, V={V}")

    rng_dtype = torch.bfloat16
    jax_dtype = jnp.bfloat16

    k, w, u, g, gk, h0, cu_seqlens = generate_inputs(B, H, K, V, seqlens_list, chunk_size, rng_dtype)

    # Prepare inputs
    k_jax = jnp.array(k.float().numpy(), dtype=jax_dtype)
    w_jax = jnp.array(w.float().numpy(), dtype=jax_dtype)
    u_jax = jnp.array(u.float().numpy(), dtype=jax_dtype)
    g_jax = jnp.array(g.float().numpy(), dtype=jax_dtype)
    gk_jax = jnp.array(gk.float().numpy(), dtype=jax_dtype)
    h0_jax = jnp.array(h0.float().numpy(), dtype=jax_dtype)
    cu_seqlens_jax = jnp.array(cu_seqlens.numpy(), dtype=jnp.int32)

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunk_indices_jax = jnp.array(chunk_indices.numpy(), dtype=jnp.int32)

    # Warmup
    print("Warming up...")
    for _ in range(args.warmup):
        h_history_jax, _, _ = pallas_fwd(
            k=k_jax, w=w_jax, v=u_jax, g=g_jax, gk=gk_jax,
            initial_state=h0_jax, output_final_state=True,
            chunk_size=chunk_size, save_new_value=True,
            seqlens=cu_seqlens_jax,
            chunk_indices=chunk_indices_jax,
            use_exp2=False
        )
        jax.block_until_ready(h_history_jax)

    # Benchmark
    print(f"Benchmarking ({args.iter} iterations)...")
    start_time = time.time()
    for _ in range(args.iter):
        h_history_jax, _, _ = pallas_fwd(
            k=k_jax, w=w_jax, v=u_jax, g=g_jax, gk=gk_jax,
            initial_state=h0_jax, output_final_state=True,
            chunk_size=chunk_size, save_new_value=True,
            seqlens=cu_seqlens_jax,
            chunk_indices=chunk_indices_jax,
            use_exp2=False
        )
        jax.block_until_ready(h_history_jax)
    end_time = time.time()

    avg_time = (end_time - start_time) / args.iter
    print(f"Average time per iteration: {avg_time*1000:.4f} ms")

    # Calculate throughput (tokens/sec)
    tokens_per_sec = TotalT / avg_time
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")


def main():
    parser = argparse.ArgumentParser(description="Test and Benchmark Pallas/Triton kernels")
    parser.add_argument('--bench', action='store_true', help="Run benchmark mode")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size (number of sequences)")
    parser.add_argument('--seq-len', type=int, default=128, help="Sequence length. Note: Large values (>512) may cause OOM on simulation backends due to full-sequence loading in Pallas kernel.")
    parser.add_argument('--num-heads', type=int, default=4, help="Number of heads")
    parser.add_argument('--head-dim', type=int, default=64, help="Head dimension (K and V)")
    parser.add_argument('--warmup', type=int, default=5, help="Warmup iterations")
    parser.add_argument('--iter', type=int, default=20, help="Benchmark iterations")

    args = parser.parse_args()

    if args.bench:
        run_benchmark(args)
    else:
        # run_correctness(args)
        run_comparison_o_gk_varlen(args)

if __name__ == "__main__":
    main()