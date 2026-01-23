# Delta Attention Comparison Benchmark

该子项目用于对比测试 `KimiDeltaAttention` 和 `Qwen3NextGatedDeltaNet` 的训练性能。

## 运行方式

### 1. 本地 (CPU/Mac/GPU)
确保已安装 `uv`，然后在当前目录下运行：
```bash
uv run benchmark_script.py
```

### 2. Google Cloud TPU VM
在 TPU VM 上，需要安装特定的 TPU 版本 JAX (`libtpu`)。请使用提供的脚本进行配置：

```bash
# 1. 运行设置脚本 (创建 venv 并安装 TPU 依赖)
bash setup_tpu.sh

# 2. 激活环境
source .venv/bin/activate

# 3. 运行测试
python benchmark_script.py
```

## 结构说明
- `benchmark_script.py`: 主测试脚本，包含前向+反向传播计时及 Profiler 埋点。会自动检测并打印当前运行的设备 (TPU/CPU)。
- `setup_tpu.sh`: 专门用于 TPU 环境的安装脚本，从 Google Storage 安装 `jax[tpu]`。
- `src/layers/`: 存放适配后的 Attention 模块代码。
- `src/`: 存放最小化的依赖实现（Linears, Normalizations 等），去除了原 MaxText 项目中的复杂依赖。
- `profiles/`: 运行后生成的 JAX Profile 文件。