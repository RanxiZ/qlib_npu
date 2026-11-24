# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NPU-adapted GRU model.

Ascend NPU adaptation of the time-series GRU model. The recurrent operator is
dispatched through ``npu_rnn.rnn_forward``, which unrolls ``nn.GRU`` into
aclnn-supported ``linear``/``sigmoid``/``tanh`` ops on NPU, and the accelerator
is resolved via ``torch_device.resolve_torch_device``. Run directly on NPU with::

    qrun examples/benchmarks/GRU/workflow_config_gru_Alpha158.yaml
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .pytorch_utils import count_parameters
from .torch_device import empty_torch_cache, resolve_torch_device, use_torch_device, is_npu_available
from .npu_rnn import rnn_forward
from .infer_utils import InferTensorCache, to_infer_device
from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from ...model.utils import ConcatDataset
from ...data.dataset.weight import Reweighter


class GRU_NPU(Model):
    """GRU Model (Ascend NPU adapted)

    Parameters
    ----------
    d_feat : int
        input dimension for each time step
    metric: str
        the evaluation metric used in early stop
    optimizer : str
        optimizer name
    GPU : str
        the GPU ID(s) used for training
    """

    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        use_fp16=False,
        pred_batch_size=16000,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("GRU_NPU")
        self.logger.info("GRU_NPU pytorch version...")

        # set hyper-parameters.
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        if not is_npu_available():
            raise RuntimeError("NPU requested but torch_npu/NPU is unavailable; refusing CPU fallback")
        self.device = resolve_torch_device(gpu=GPU)
        if self.device.type != "npu":
            raise RuntimeError(f"{self.__class__.__name__} requires NPU, got {self.device}")
        self._backend_confirmed = False
        self.n_jobs = n_jobs
        self.seed = seed
        self.use_fp16 = use_fp16
        self.pred_batch_size = pred_batch_size

        # cache for the last uploaded inference features (data "常驻 NPU")
        self._infer_cache = InferTensorCache()

        self.logger.info(
            "GRU_NPU parameters setting:"
            "\nd_feat : {}"
            "\nhidden_size : {}"
            "\nnum_layers : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\ndevice : {}"
            "\nn_jobs : {}"
            "\nuse_GPU : {}"
            "\nseed : {}".format(
                d_feat,
                hidden_size,
                num_layers,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                self.device,
                n_jobs,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.GRU_model = GRUNPUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.logger.info("model:\n{:}".format(self.GRU_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.GRU_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.GRU_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.GRU_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.GRU_model.to(self.device)

    @property
    def use_gpu(self):
        return use_torch_device(self.device)

    def mse(self, pred, label, weight):
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, weight=None):
        mask = ~torch.isnan(label)

        if weight is None:
            weight = torch.ones_like(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):
        self.GRU_model.train()

        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.GRU_model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.GRU_model.parameters(), 3.0)
            self.train_optimizer.step()
            if not self._backend_confirmed:
                params = list(self.GRU_model.parameters())
                if any(p.device != self.device for p in params) or feature.device != self.device or pred.device != self.device:
                    raise RuntimeError("NPU parameter/input/output device mismatch")
                grads = [p.grad for p in params if p.grad is not None]
                if not grads or any(g.device != self.device or not bool(torch.isfinite(g).all()) for g in grads):
                    raise RuntimeError("NPU gradients missing, on wrong device, or nonfinite")
                self.logger.info("NPU TRAINING CONFIRMED: device=%s backend=unrolled_gru first_optimizer_step_complete=true", self.device)
                self._backend_confirmed = True

    def test_epoch(self, data_loader):
        self.GRU_model.eval()

        scores = []
        losses = []

        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            # feature[torch.isnan(feature)] = 0
            label = data[:, -1, -1].to(self.device)

            with torch.no_grad():
                pred = self.GRU_model(feature.float())
                loss = self.loss_fn(pred, label, weight.to(self.device))
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=True,
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.GRU_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GRU_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            empty_torch_cache(self.device)
        self.logger.info("NPU TRAINING COMPLETE: device=%s epochs=%d best_epoch=%d", self.device, step + 1, best_epoch)

    def predict(self, dataset, segment="test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        index = dl_test.get_index()
        self.GRU_model.eval()

        # Materialize the whole test feature set in one vectorized call (no
        # per-sample Python loop / DataLoader overhead) and upload it once to
        # the accelerator; features stay resident across repeated ``predict``.
        feature = dl_test.to_numpy()[:, :, 0:-1]  # [N, step_len, d_feat]
        data = to_infer_device(feature, self.device, use_fp16=self.use_fp16, cache=self._infer_cache)

        preds = []
        with torch.no_grad():
            for begin in range(0, len(data), self.pred_batch_size):
                x = data[begin : begin + self.pred_batch_size]
                preds.append(self.GRU_model(x).detach())
        preds = torch.cat(preds).cpu().numpy()

        return pd.Series(preds.reshape(-1), index=index)


class GRUNPUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        out = rnn_forward(self.rnn, x)
        return self.fc_out(out[:, -1, :]).squeeze(-1)
