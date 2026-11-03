# GRU / ALSTM NPU 算子替换报告（vs qlib 源码）

本报告对照说明：为适配华为 Ascend NPU（torch_npu），`GRU` / `ALSTM` 模型相较
qlib 上游源码（`microsoft/qlib`）在**算子层面**做了哪些替换、替换原因，以及数值
等价性验证结果。

---

## 1. 背景：为什么必须替换算子

Ascend NPU 上的 `torch_npu` 只支持 **aclnn** 算子。qlib 源码中 RNN 模型直接调用
PyTorch 原生循环网络算子：

- `GRU`：`nn.GRU` → `torch._VF.gru` → `aten::gru`
- `ALSTM`：`nn.LSTM` / `nn.GRU` → `aten::lstm` / `aten::gru`

在 NPU 上直接调用会报错：

```
RuntimeError: Current device only support aclnn operator,
but current operator gru do not have aclnn implementation
```

因此需要把原生 RNN 算子替换为**由 aclnn 已支持的基础算子组合成的等价实现**。

---

## 2. 算子替换清单

| 模型 | qlib 源码算子 | NPU 替换算子 | 替换方式 | 触发条件 |
|------|--------------|-------------|---------|---------|
| GRU | `aten::gru`（`nn.GRU`，`_VF.gru`） | `aten::linear` + `aten::sigmoid` + `aten::tanh` + `aten::add/mul/sub` + `aten::stack`（逐时间步展开） | `npu_rnn.py::gru_layer_forward` | `x.device.type == "npu"` |
| ALSTM(`rnn_type=LSTM`) | `aten::lstm`（`nn.LSTM`，`_VF.lstm`） | 同上基本算子（逐时间步展开 LSTM cell） | `npu_rnn.py::lstm_layer_forward` | `x.device.type == "npu"` |
| ALSTM/GRU 其他设备 | `aten::gru` / `aten::lstm` | 保持原生 `rnn(x)` 不变 | `npu_rnn.py::rnn_forward` 的 else 分支 | 其他 device |

> 关键点：`rnn_forward()` 在**非 NPU 设备上仍然走原生算子**，只在 `npd` 设备上
> 才展开为基础算子，因此不影响 CUDA/CPU 上的行为。

---

## 3. 展开后的算子实现

### 3.1 GRU cell（`gru_layer_forward`）

门控顺序为 `r, z, n`（与 `nn.GRU` 权重前导维一致）。

```
x_gates = linear(x, W_ih, b_ih)          # [B, S, 3H]
for t in range(seq_len):
    hg      = linear(h, W_hh, b_hh)      # [B, 3H]
    r       = sigmoid(g[:, :H]   + hg[:, :H])
    z       = sigmoid(g[:, H:2H] + hg[:, H:2H])
    n       = tanh(g[:, 2H:]     + r * hg[:, 2H:])
    h       = (1 - z) * n + z * h
outs = stack(hs, dim=1)                   # [B, S, H]
```

### 3.2 LSTM cell（`lstm_layer_forward`）

门控顺序为 `i, f, g, o`（与 `nn.LSTM` 权重前导维一致）。

```
x_gates = linear(x, W_ih, b_ih)          # [B, S, 4H]
for t in range(seq_len):
    hg      = linear(h, W_hh, b_hh)      # [B, 4H]
    i       = sigmoid(g[:, :H]     + hg[:, :H])
    f       = sigmoid(g[:, H:2H]   + hg[:, H:2H])
    gg      = tanh(g[:, 2H:3H]     + hg[:, 2H:3H])
    o       = sigmoid(g[:, 3H:]    + hg[:, 3H:])
    c       = f * c + i * gg
    h       = o * tanh(c)
outs = stack(hs, dim=1)                   # [B, S, H]
```

### 3.3 优缺点

- ✅ 优点：全部由 NPU 已支持的基础算子组成，**数值上等价于原生算子**（至 fp32
  舍入误差），可直接复用 `nn.GRU`/`nn.LSTM` 的权重，无需改模型结构。
- ⚠️ 缺点：按时间步展开，训练/推理吞吐量低于高度优化的原生 CUDA/AscendC kernel
  （本报告不涉及 AscendC 自定义算子方案，那是 `qlib_AscendC_GRU*.sh` 的另一条路线）。

---

## 4. 推理数据搬运优化（`infer_utils.py`）

严格来说不属于“算子”替换，但与之配套、影响 NPU 推理正确性与性能：

| 项目 | qlib 源码 | NPU 版本 |
|------|-----------|---------|
| 特征上传 | `DataFrame.values`（F 序）逐 batch `torch.from_numpy().to(device)` | 整表一次上传；F 序按 `ravel(order="F")` 零拷贝 flat 上传后在设备端转置 |
| 分批 | 按 `self.batch_size` 遍历 Python 切片 | 常驻设备张量上按 `pred_batch_size` 切片 |
| 精度 | fp32 | 可选 `use_fp16`（host→device 半字节传输，设备端转回 fp32，精度 ~1e-3） |
| 重复预测 | 每次重新上传 | `InferTensorCache`（弱引用）缓存上次特征张量 |

---

## 5. 涉及文件

新增：

- `qlib/contrib/model/npu_rnn.py`：GRU/LSTM cell 展开实现 + `rnn_forward` 分发
- `qlib/contrib/model/infer_utils.py`：`to_infer_device` / `InferTensorCache` / `predict_in_batches`
- `qlib/contrib/model/torch_device.py`：`resolve_torch_device`（cuda/npu/cpu 抽象）
- `qlib/contrib/model/accelerator_config.py`：`qlib_init.accelerator` 合并

修改：

- `qlib/contrib/model/pytorch_gru.py` / `pytorch_gru_ts.py`
- `qlib/contrib/model/pytorch_alstm.py` / `pytorch_alstm_ts.py`
- `qlib/config.py`（`accelerator` 默认项）、`qlib/model/trainer.py`（注入 GPU）
- `qlib/data/dataset/__init__.py`（`TSDataSampler.to_numpy` 向量化）
- `examples/benchmarks/GRU/workflow_config_gru_Alpha158.yaml`
- `examples/benchmarks/ALSTM/workflow_config_alstm_Alpha158.yaml`

---

## 6. 数值等价验证（Ascend950PR，torch 2.10 + torch_npu 2.10）

以 CPU 上原生 `nn.GRU`/`nn.LSTM` 输出为参考，NPU 上 `rnn_forward` 展开实现对比：

| 用例 | max_abs_diff | allclose(atol/rtol=1e-4) |
|------|-------------|--------------------------|
| GRU L=1 单向 | 1.19e-07 | ✅ |
| GRU L=2 单向 | 5.96e-08 | ✅ |
| LSTM L=1 单向 | 8.94e-08 | ✅ |
| LSTM L=2 单向 | 8.94e-08 | ✅ |
| `GRUModel` 前向 | 4.47e-08 | ✅ |
| `ALSTMModel(GRU)` 前向 | 2.98e-08 | ✅ |
| `ALSTMModel(LSTM)` 前向 | 3.73e-08 | ✅ |

端到端（csi300 小样本、1 epoch）在 NPU 上 `fit` + `predict` 均正常跑通，重复
`predict`（缓存命中）结果一致。

---

## 7. 结论

GRU/ALSTM 迁移到 NPU 在**算子层**的改动集中在 RNN 前向算子：用
`linear + sigmoid + tanh + pointwise` 的组合展开等价替换原生
`aten::gru` / `aten::lstm`，并在非 NPU 设备上保持原生算子不变。验证结果表明两者
数值等价（误差在 fp32 舍入量级内），模型可正常训练与推理。