# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gru_layer_forward(x: torch.Tensor, w_ih, w_hh, b_ih, b_hh) -> torch.Tensor:
    """Single GRU layer forward over a whole sequence.

    Mathematically identical to ``nn.GRU`` (gate order ``r, z, n`` along the
    leading weight dimension). Implemented with plain ``linear`` / ``sigmoid`` /
    ``tanh`` and pointwise ops, which are all supported on Ascend NPU, because
    the native ``torch.nn.GRU`` (``gru`` / ``_VF.gru``) has no aclnn
    implementation on torch_npu.
    """
    batch, seq_len, _ = x.shape
    hidden = w_hh.shape[0] // 3
    # Input projection is time-independent and can be computed at once.
    x_gates = F.linear(x, w_ih, b_ih)  # [B, S, 3H]
    h = x.new_zeros(batch, hidden)
    outs = []
    for t in range(seq_len):
        g = x_gates[:, t, :]  # [B, 3H]
        hg = F.linear(h, w_hh, b_hh)  # [B, 3H]
        r = torch.sigmoid(g[:, :hidden] + hg[:, :hidden])
        z = torch.sigmoid(g[:, hidden : 2 * hidden] + hg[:, hidden : 2 * hidden])
        n = torch.tanh(g[:, 2 * hidden :] + r * hg[:, 2 * hidden :])
        h = (1 - z) * n + z * h
        outs.append(h)
    return torch.stack(outs, dim=1)  # [B, S, H]


def lstm_layer_forward(x: torch.Tensor, w_ih, w_hh, b_ih, b_hh) -> torch.Tensor:
    """Single LSTM layer forward over a whole sequence.

    Mathematically identical to ``nn.LSTM`` (gate order ``i, f, g, o`` along the
    leading weight dimension). Used so the ALSTM model also runs on NPU when
    ``rnn_type=LSTM``, since ``torch.nn.LSTM`` and ``nn.LSTMCell`` both lack an
    aclnn implementation on torch_npu.
    """
    batch, seq_len, _ = x.shape
    hidden = w_hh.shape[0] // 4
    x_gates = F.linear(x, w_ih, b_ih)  # [B, S, 4H]
    h = x.new_zeros(batch, hidden)
    c = x.new_zeros(batch, hidden)
    outs = []
    for t in range(seq_len):
        g = x_gates[:, t, :]  # [B, 4H]
        hg = F.linear(h, w_hh, b_hh)  # [B, 4H]
        i = torch.sigmoid(g[:, :hidden] + hg[:, :hidden])
        f = torch.sigmoid(g[:, hidden : 2 * hidden] + hg[:, hidden : 2 * hidden])
        gg = torch.tanh(g[:, 2 * hidden : 3 * hidden] + hg[:, 2 * hidden : 3 * hidden])
        o = torch.sigmoid(g[:, 3 * hidden :] + hg[:, 3 * hidden :])
        c = f * c + i * gg
        h = o * torch.tanh(c)
        outs.append(h)
    return torch.stack(outs, dim=1)  # [B, S, H]


def rnn_forward(rnn: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """RNN forward that returns the full output sequence ``[B, S, H]``.

    On Ascend NPU the native ``nn.GRU``/``nn.LSTM`` operators have no aclnn
    implementation, so the RNN is unrolled into per-cell ``linear`` +
    ``sigmoid``/``tanh`` computations (bitwise-compatible with the reference
    module up to fp32 rounding). On other devices it dispatches straight to the
    native module.
    """
    if x.device.type == "npu":
        if isinstance(rnn, nn.GRU):
            out = x
            for layer in range(rnn.num_layers):
                if layer > 0 and rnn.dropout > 0 and rnn.training:
                    out = F.dropout(out, p=rnn.dropout)
                out = gru_layer_forward(
                    out,
                    getattr(rnn, f"weight_ih_l{layer}"),
                    getattr(rnn, f"weight_hh_l{layer}"),
                    getattr(rnn, f"bias_ih_l{layer}"),
                    getattr(rnn, f"bias_hh_l{layer}"),
                )
            return out
        if isinstance(rnn, nn.LSTM):
            out = x
            for layer in range(rnn.num_layers):
                if layer > 0 and rnn.dropout > 0 and rnn.training:
                    out = F.dropout(out, p=rnn.dropout)
                out = lstm_layer_forward(
                    out,
                    getattr(rnn, f"weight_ih_l{layer}"),
                    getattr(rnn, f"weight_hh_l{layer}"),
                    getattr(rnn, f"bias_ih_l{layer}"),
                    getattr(rnn, f"bias_hh_l{layer}"),
                )
            return out
    output, _ = rnn(x)
    return output
