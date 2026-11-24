# GRU on Ascend NPU

Requires an Ascend NPU with compatible driver, firmware, CANN, PyTorch and torch_npu. Source the installed CANN set_env.sh first. CPU-only hosts cannot run this configuration.

Clone branch qlib_npu_GRU into /workspace/qlib. In the active NPU Python environment run:

    cd /workspace/qlib
    python -m pip install -e .
    python -c 'import qlib, torch, torch_npu; print(qlib.__file__); assert torch.npu.is_available()'

Prepare China daily data at ~/.qlib/qlib_data/cn_data, or change provider_uri in the YAML. With network access:

    python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

Do not install the historical requirements.txt pins: torch 1.7 is not the validated Ascend runtime. Use torch and torch_npu compatible with your CANN installation.

Run the complete workflow:

    cd /workspace/qlib/examples/
    qrun benchmarks/GRU/workflow_config_gru_Alpha158.yaml

This selects GRU_NPU on npu:0 and includes training, prediction, signal analysis and portfolio backtest. Missing NPU support raises an error instead of silently falling back to CPU. Wait for other NPU jobs to finish before benchmarking.

Default seed is None. For the timing comparison, add seed: 20261123 under task.model.kwargs and set PYTHONHASHSEED=20261123 before starting qrun.

On 2026-11-23, Ascend950PR, identical runtime/data/seed, sequential complete qrun processes: branch GRU 381.601905 seconds; GRU_NPU 379.220449 seconds. Both exited 0 after 15 epochs. These are single observations, not guaranteed speedups. Exact times and metrics depend on hardware, software, data and seed. Alpha360 is unchanged; this change targets Alpha158.
