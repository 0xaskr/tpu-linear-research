import time
import functools
import sys
import csv
import os
import subprocess
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pl_tpu
from scipy import stats
import argparse

def run_benchmark(loop_counts, workload_per_iter=1, dim_m=2048, dim_n=2048, dim_k=2048, dtype=jnp.dtype, acc_dtype=jnp.dtype):
    def kernel(x_ref, w_ref, out_ref, *, loop_count):
        # Use a fori_loop whose carry is the accumulated matmul result; x_ref/w_ref come from the closure.
        def body(i, _):
            return jnp.dot(
                x_ref[...], w_ref[...], preferred_element_type=acc_dtype
            )

        init = jnp.zeros((dim_m, dim_n), dtype=acc_dtype)
        result = jax.lax.fori_loop(0, loop_count, body, init)
        # Cast back to the original dtype for the output buffer.
        out_ref[...] = result.astype(x_ref.dtype)

    @functools.partial(jax.jit, static_argnums=(2,))
    def run(x, w, loop_count):
        bound_kernel = functools.partial(kernel, loop_count=loop_count)

        return pl.pallas_call(
            bound_kernel,
            out_shape=jax.ShapeDtypeStruct((dim_m, dim_n), x.dtype),
            interpret=False,
            grid=(),
            compiler_params=pl_tpu.CompilerParams(
                vmem_limit_bytes=96 * 1024 * 1024,
                disable_bounds_checks=True,
            ),
        )(x, w)

    print(f"\n=== Benchmarking =====")

    x = jnp.zeros((dim_m, dim_k), dtype=dtype)
    w = jnp.zeros((dim_k, dim_n), dtype=dtype)
    jax.block_until_ready(x)
    jax.block_until_ready(w)

    times_ns = []
    for n in loop_counts:
        print("  -> Warming up...")
        tw_0 = time.perf_counter_ns()
        _ = run(x, w, n).block_until_ready()
        tw_1 = time.perf_counter_ns()
        print(f"  -> Warmed up in ({(tw_1-tw_0)/1e3:.2f} us)")

        t0 = time.perf_counter_ns()
        out = run(x, w, n)
        out.block_until_ready()
        t1 = time.perf_counter_ns()

        duration = t1 - t0
        times_ns.append(duration)
        print(f"  -> N={n:<6} Time={duration/1e3:.2f} us")

    slope, intercept, r_value, _, _ = stats.linregress(loop_counts, times_ns)

    print(f"\n[Analysis Result for matmul]")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  1. Base Overhead (Intercept): {intercept/1e3:.2f} us")

    if slope > 0:
        throughput_ops_per_ns = workload_per_iter / slope
        throughput_tflops = throughput_ops_per_ns / 1000.0

        print(f"  2. Per-Iter Latency (Slope):  {slope:.4f} ns")
        print(f"  3. Measured Throughput:       {throughput_ops_per_ns:.2f} Ops/ns")
        print(f"                                ({throughput_tflops:.2f} TFLOPS equivalent)")
    else:
        print("  [Error] Slope is negative or zero.")

    return intercept, slope


def parse_args():
    parser = argparse.ArgumentParser(description="TPU Matmul Benchmark")
    
    parser.add_argument("--m", type=int, default=2048, help="Total M")
    parser.add_argument("--n", type=int, default=2048, help="Total N")
    parser.add_argument("--k", type=int, default=2048, help="Total K")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Dtype")
    parser.add_argument("--batch", action="store_true", help="Run batch benchmark from CSV files")
    
    return parser.parse_args()

def run_batch_mode():
    csv_files = ["matmul-shape/test_case_normal.csv", "matmul-shape/test_case_padding.csv"]
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping.")
            continue
            
        output_csv = csv_file.replace(".csv", "_results.csv")
        print(f"\nProcessing {csv_file} -> {output_csv}...")
        
        batch_results = []
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    m = int(row['M'])
                    n = int(row['N'])
                    k = int(row['K'])
                    dtype_str = row['Dtype']
                    
                    if dtype_str.upper() == "BF16":
                        dtype_arg = "bfloat16"
                    else:
                        dtype_arg = dtype_str.lower()
                    
                    prefix = "normal" if "normal" in csv_file else "padding"
                    dump_dir = f"/tmp/mosaic_dumps/{prefix}_{m}_{n}_{k}_{dtype_arg}"
                    os.makedirs(dump_dir, exist_ok=True)
                    
                    env = os.environ.copy()
                    existing_args = env.get("LIBTPU_INIT_ARGS", "")
                    new_args = f"{existing_args} --xla_mosaic_dump_to={dump_dir}".strip()
                    env["LIBTPU_INIT_ARGS"] = new_args
                    
                    print(f"Running case: M={m}, N={n}, K={k}, Dtype={dtype_arg}, DumpTo={dump_dir}")
                    
                    # Capture output to parse results
                    result = subprocess.run(
                        [sys.executable, __file__, "--m", str(m), "--n", str(n), "--k", str(k), "--dtype", dtype_arg],
                        env=env,
                        capture_output=True,
                        text=True
                    )
                    
                    # Echo stdout to console so user sees progress
                    print(result.stdout)
                    if result.stderr:
                        print(result.stderr, file=sys.stderr)
                    
                    if result.returncode != 0:
                         print(f"Benchmark failed for {row} with return code {result.returncode}")
                         continue

                    # Parse results from stdout
                    slope = 0.0
                    compute_metric = 0.0
                    
                    for line in result.stdout.splitlines():
                        if line.startswith("BENCH_RESULT:"):
                            try:
                                parts = line.split(":")[1].split(",")
                                slope = float(parts[0])
                                if len(parts) > 1:
                                    compute_metric = float(parts[1])
                            except (IndexError, ValueError):
                                print("Failed to parse BENCH_RESULT line")
                    
                    if compute_metric == 0.0 and slope > 0:
                         # Fallback calculation, should also be scaled
                         compute_metric = ((m * n * n * k) / slope) / 10000.0

                    batch_results.append({
                        "M": m,
                        "N": n,
                        "K": k,
                        "Dtype": dtype_str,
                        "Per-Iter Latency (ns)": slope,
                        "Compute Metric": compute_metric
                    })
                    
                except (ValueError, KeyError) as e:
                    print(f"Skipping invalid row: {row} - Error: {e}")
        
        # Write results to new CSV
        if batch_results:
            fieldnames = ["M", "N", "K", "Dtype", "Per-Iter Latency (ns)", "Compute Metric"]
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(batch_results)
            print(f"Written {len(batch_results)} results to {output_csv}")
        else:
            print(f"No results generated for {csv_file}")


if __name__ == "__main__":

    args = parse_args()

    if args.batch:
        run_batch_mode()
        sys.exit(0)

    dtype = jnp.bfloat16
    if args.dtype == "float32":
        dtype = jnp.float32
    elif args.dtype == "int8":
        dtype = jnp.int8
    
    print(f"Device: {jax.devices()[0]}")
    print(f"matmul: <{args.m}, {args.k}> x <{args.k}, {args.n}>")
    print(f"dtype: {dtype}")

    matmul_flops = 2 * (args.m * args.n * args.k)

    intercept, slope = run_benchmark(
        [5000, 10000, 20000],
        workload_per_iter=matmul_flops,
        dim_m=args.m,
        dim_n=args.n,
        dim_k=args.k,
        dtype=dtype,
        acc_dtype=jnp.int32 if dtype == jnp.int8 else jnp.float32
    )
    
    raw_compute_metric = (args.m * args.n * args.k) / slope if slope > 0 else 0.0
    scaled_compute_metric_for_output = raw_compute_metric / 10000.0
    # Print machine-readable result for batch mode parsing
    print(f"BENCH_RESULT:{slope:.2f},{scaled_compute_metric_for_output:.2f}")