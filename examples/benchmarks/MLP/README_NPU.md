# MLP NPU 原版实现与复现

本分支的 `pytorch_nn.py` 同步自 HiDeVLab 成功运行的原安装包，不混入其他模型的定制修改。包括 NPU 设备选择/缓存释放、训练数据设备驻留、推理数据缓存、Fortran 布局上传及可选 fp16 传输。设备解析工具 `torch_device.py` 已在分支中且与原安装包相同。

与此前分支实现相比：恢复原版训练张量布局；恢复模型内 `_to_device` 弱引用缓存；推理 batch 恢复固定 8096，不采用此前分支默认 16000。`use_fp16=False` 保持默认，不等同于启用 AMP。

## 环境准备（一次）

不是只 clone 就能在任意机器运行：须有 Ascend NPU、匹配的驱动/CANN/PyTorch/torch_npu、Qlib 依赖、编译工具，以及完整 CN 行情数据。推荐隔离环境，避免已有 site-packages 版本掩盖源码。

HiDeVLab 参考版本：Python 3.12.13、CANN 9.1.0、torch 2.10.0+cpu、torch_npu 2.10.0.post4、numpy 2.0.0、pandas 2.3.3、mlflow 3.12.0、filelock 3.29.7、scikit-learn 1.9.1。torch 的 +cpu 标签不表示 NPU 不可用，NPU 后端由 torch_npu 提供。安装匹配的框架包请遵循设备供应商说明。

```bash
git clone --branch qlib_npu_MLP --single-branch https://github.com/RanxiZ/qlib_npu.git /workspace/qlib
cd /workspace/qlib
# 在已有配套 torch/torch_npu 与 Qlib 依赖的环境中执行；不升级系统依赖
MLP_PYTHON=/usr/local/python3.12.13/bin/python3.12 \
  bash examples/benchmarks/MLP/setup_mlp_npu.sh
source /workspace/qlib/.venv-mlp-npu/bin/activate
```

若 CANN 路径不同，执行脚本时设置 `CANN_ENV`。构建需要 setuptools、setuptools-scm、Cython、NumPy 和 C++ 编译器。脚本采用 `--system-site-packages` 复用已有硬件依赖，并以 editable 安装确保加载本分支。未安装完整依赖时先按仓库安装说明准备；本脚本不联网升级依赖。

YAML 数据路径为 `~/.qlib/qlib_data/cn_data`，必须准备覆盖 2008-01-01 至 2020-08-01、包含 CSI300 和 SH000300 的完整数据。不同数据快照会改变结果，不能仅凭相同 YAML 声称数值一致。已有 `/workspace/qlib` 时请先备份或使用独立目录，不覆盖未提交修改。

## 正常运行

在已激活上述环境的终端：

```bash
cd /workspace/qlib/examples/ && \
qrun benchmarks/MLP/workflow_config_mlp_Alpha158.yaml
```

包装器自动加载 CANN 后调用原 Qlib CLI。YAML 指定 NPU:0，157 维特征，MSE/Adam，batch_size=8192、max_steps=8000，并执行信号、信号分析和回测三个 Record。max_steps 为上限，实际步数受早停影响。

若实际导入路径不是本分支，请检查 `command -v qrun` 和 Python 环境，不能只凭工作目录认定源码已生效。

## 已有运行证据与边界

2026-11-16 HiDeVLab 原安装包使用该模型和 YAML 完整 qrun 成功，退出码 0，进程端到端 136.795939 秒。该数值是一次观测，不承诺其他机器相同耗时。该次模型 SHA256：`9d00b2f4ae4213bc708fb85a7a4c718c068f38c6b74b14e6247c493c7e2de424`。

精度校准和 950/A100 端到端耗时比约 0.4426 是后续目标，尚未验收。默认未固定训练 seed，重跑不保证逐位相同。

隔离分支复验：2026-11-16，安装脚本成功，实际加载本分支源码；完整 qrun 退出码 0，进程端到端 139.260100 秒。日志保存在 HiDeVLab /workspace/qlib_mlp_switch_20261116/branch_qrun.log。原安装包保持不变。
