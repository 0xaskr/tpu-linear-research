import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import functools

from src.layers.kimi_delta_attention import KimiDeltaAttention
from src.layers.qwen3 import Qwen3NextGatedDeltaNet
from src.common_types import Config

# Dummy Config
class BenchmarkConfig:
    def __init__(self):
        self.emb_dim = 2048
        self.hidden_size = 2048
        self.gdn_num_value_heads = 16
        self.gdn_num_key_heads = 16
        self.gdn_value_head_dim = 128
        self.gdn_key_head_dim = 128
        self.gdn_conv_kernel_dim = 4
        self.gdn_chunk_size = 256
        self.dtype = jnp.bfloat16 # Use bfloat16 for TPU/GPU benchmarking
        self.weight_dtype = jnp.bfloat16
        self.normalization_layer_epsilon = 1e-6
        self.use_qk_norm_in_gdn = True
        self.matmul_precision = "default"

config = BenchmarkConfig()

# Check and print JAX devices
print("-" * 40)
print(f"JAX Default Backend: {jax.default_backend()}")
devices = jax.local_devices()
print(f"Available Devices ({len(devices)}): {devices}")
if any("TPU" in str(d) for d in devices):
    print("🚀 Running on TPU!")
else:
    print("⚠️  TPU not detected. Running on CPU/GPU.")
print("-" * 40)

# Parameters
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
SEQ_LENS = [1024, 2048, 4096, 8192]
WARMUP_STEPS = 2
STEPS = 3

def benchmark_layer(layer_cls, layer_name, create_args, input_shape, steps=STEPS, warmup=WARMUP_STEPS):
    rngs = nnx.Rngs(0)
    
    # Create model
    if layer_name == "KimiDeltaAttention":
        model = layer_cls(**create_args, rngs=rngs)
    else:
        model = layer_cls(create_args, rngs=rngs) # Qwen3 takes config as first arg

    optimizer = nnx.Optimizer(model, optax.adamw(1e-4), wrt=nnx.Param)

    # Create dummy input
    key = jax.random.PRNGKey(0)
    try:
        x = jax.random.normal(key, input_shape, dtype=config.dtype)
    except Exception as e:
        raise RuntimeError(f"Input creation failed: {e}")

    @nnx.jit
    def train_step(model, optimizer, x):
        def loss_fn(model):
            out = model(x)
            if isinstance(out, tuple): # Kimi returns (out, state)
                out = out[0]
            return jnp.mean(out)
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # Warmup
    for _ in range(warmup):
        train_step(model, optimizer, x).block_until_ready()

    # Benchmark
    start_time = time.perf_counter()
    
    for i in range(steps):
        train_step(model, optimizer, x).block_until_ready()
        
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / steps
    return avg_time

def main():
    # Ensure profile directory exists
    import os
    os.makedirs("profiles", exist_ok=True)

    # Kimi Args Template
    kimi_args_template = {
        "hidden_size": config.hidden_size,
        "num_heads": config.gdn_num_value_heads, # Assume v_heads = num_heads
        "head_dim": config.gdn_value_head_dim,
        "conv_kernel_size": config.gdn_conv_kernel_dim,
        "normalization_layer_epsilon": config.normalization_layer_epsilon,
        "dtype": config.dtype,
        "weight_dtype": config.weight_dtype,
    }

    print(f"{'Batch':<6} | {'Seq':<6} | {'Kimi (ms)':<10} | {'Qwen (ms)':<10} | {'Diff (%)':<10}")
    print("-" * 55)

    for b in BATCH_SIZES:
        for s in SEQ_LENS:
            input_shape = (b, s, config.emb_dim)
            
            try:
                t_kimi = benchmark_layer(KimiDeltaAttention, "KimiDeltaAttention", kimi_args_template, input_shape)
                t_qwen = benchmark_layer(Qwen3NextGatedDeltaNet, "Qwen3NextGatedDeltaNet", config, input_shape)
                
                diff = (t_qwen - t_kimi) / t_kimi * 100
                print(f"{b:<6} | {s:<6} | {t_kimi*1000:<10.2f} | {t_qwen*1000:<10.2f} | {diff:<10.2f}")
            except Exception as e:
                import traceback
                import sys
                traceback.print_exc()
                err_msg = str(e)
                if "Resource exhausted" in err_msg or "OOM" in err_msg:
                    print(f"{b:<6} | {s:<6} | {'OOM':<10} | {'OOM':<10} | {'-':<10}")
                else:
                    print(f"{b:<6} | {s:<6} | {'Err':<10} | {'Err':<10} | {err_msg[:20]}")
                    sys.exit(1)

if __name__ == "__main__":
    main()