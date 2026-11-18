# Multi-Layer Perceptron (MLP)

## 从命令行指定 MLP Alpha158 的 seed

在 HiDeVLab 使用已部署的 CUDA 对齐入口：

```bash
cd /workspace/qlib/examples
time MLP_SEED=20261116 PYTHONHASHSEED=20261116 qrun benchmarks/MLP/workflow_config_mlp_Alpha158_seed_env.yaml
```

`MLP_SEED` 由 qrun 渲染进 `task.model.kwargs.seed`，控制初始化、批次采样和 Dropout。启动 Python 前设置 `PYTHONHASHSEED`，可固定回测中的集合遍历顺序。复现对照结果时，两端使用相同的两个值。逐位验收覆盖 `20261116` 和 `20261117`。更换 seed 可能改变早停步数和端到端耗时。

A100 使用同名模板和命令，先切换到 `/home/h60072769/qlib/examples`。从该机器的原始 CUDA YAML 生成模板，在 `task.model.kwargs` 下添加 `seed: {{ MLP_SEED | default("null") }}`，并保留 CUDA 的设备设置。

在新的 Ascend NPU 环境中，先在仓库根目录运行：

```bash
bash examples/benchmarks/MLP/setup_mlp_npu.sh
source .venv-mlp-npu/bin/activate
```

如果系统默认 `python3` 缺少 `venv` / `ensurepip`，或者 NPU 依赖装在另一套 Python 中，请通过 `MLP_PYTHON` 指定具备这些依赖的解释器。HiDeVLab 的示例：

```bash
MLP_PYTHON=/usr/local/python3.12.13/bin/python3.12 bash examples/benchmarks/MLP/setup_mlp_npu.sh
source .venv-mlp-npu/bin/activate
```

安装脚本使用已有的 CANN、PyTorch、torch-npu 和 Qlib 依赖。逐位复现还需要带 Ascend 后端的 Triton 3.2.0，以及相同的 Alpha158 CN 数据；默认数据目录为 `~/.qlib/qlib_data/cn_data`。可以用 `CANN_ENV` 指定其他 CANN 安装的初始化脚本。激活后使用上述 qrun 命令即可，仓库内的配置会启用对齐实现。

其他 seed 可通过相同命令传入，支持 0 到 4294967295 的整数。逐位一致的验收结论仅覆盖上面列出的两个 seed、当前模型结构及数据形状。A100 对照配置从原 CUDA YAML 生成，使用同样的两个环境变量。

本分支的配置会在 qrun 中显式启用 MLP 对齐逻辑。该路径复现 CUDA 的 TF32 矩阵计算、归约、Dropout 随机流、Adam 和验证 IC，依赖支持 Ascend 的 Triton。原有 MLP 配置仍可用来运行常规 NPU 实现。
