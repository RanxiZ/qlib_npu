# TCN
* Code: [https://github.com/locuslab/TCN](https://github.com/locuslab/TCN)
* Paper: [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271).


## Alpha158 on Ascend NPU

Prerequisites: Ascend device/driver, CANN, compatible PyTorch and torch_npu,
and Qlib cn_data at ~/.qlib/qlib_data/cn_data. Install in the environment
that will run qrun:

    git clone --branch qlib_npu_TCN https://github.com/RanxiZ/qlib_npu.git /workspace/qlib
    cd /workspace/qlib
    python -m pip install .
    cd /workspace/qlib/examples
    qrun benchmarks/TCN/workflow_config_tcn_Alpha158.yaml

NPU qrun automatically sources CANN and restarts Python once to initialize
the dynamic loader. Custom installation: set QLIB_CANN_SET_ENV to set_env.sh.
No source edits or manual sourcing are required. Use the same Python
environment for installation and qrun.

Original workflow verified with CANN 9.1.0: training, prediction and
portfolio backtest completed (exit 0) in 468.71 seconds for one run.
Timing depends on hardware, dataset and early stopping. Model settings
retain the original 200-epoch limit and early_stop=20.

Automatic initialization verified from examples with CANN environment variables
cleared: qrun benchmarks/TCN/workflow_config_tcn_Alpha158.yaml completed
training, prediction and backtest with exit 0 in 498.63 seconds (2026-11-17).
