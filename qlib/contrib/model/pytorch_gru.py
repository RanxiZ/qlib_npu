# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division
from __future__ import print_function
import copy
from typing import Text, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from qlib.workflow import R

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...model.base import Model
from ...utils import get_or_create_path
from .pytorch_utils import count_parameters
from .torch_device import empty_torch_cache, resolve_torch_device, use_torch_device
from .npu_rnn import rnn_forward
from .infer_utils import InferTensorCache, to_infer_device


class GRU(Model):
    """GRU Model

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
        GPU=0,
        seed=None,
        use_fp16=False,
        pred_batch_size=16000,
        **kwargs,
    ):
        # Set logger.
        self.logger = get_module_logger("GRU")
        self.logger.info("GRU pytorch version...")

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
        self.device = resolve_torch_device(gpu=GPU)
        self.seed = seed
        self.use_fp16 = use_fp16
        self.pred_batch_size = pred_batch_size

        # cache for the last uploaded inference features (data "常驻 NPU")
        self._infer_cache = InferTensorCache()

        self.logger.info(
            "GRU parameters setting:"
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
            "\nvisible_GPU : {}"
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
                GPU,
                self.use_gpu,
                seed,
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.gru_model = GRUModel(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.logger.info("model:\n{:}".format(self.gru_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.gru_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.gru_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.gru_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.gru_model.to(self.device)

    @property
    def use_gpu(self):
        return use_torch_device(self.device)

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.gru_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.gru_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.gru_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.gru_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            with torch.no_grad():
                pred = self.gru_model(feature)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        # prepare training and validation data
        dfs = {
            k: dataset.prepare(
                k,
                col_set=["feature", "label"],
                data_key=DataHandlerLP.DK_L,
            )
            for k in ["train", "valid"]
            if k in dataset.segments
        }
        df_train, df_valid = dfs.get("train", pd.DataFrame()), dfs.get("valid", pd.DataFrame())

        # check if training data is empty
        if df_train.empty:
            raise ValueError("Empty training data from dataset, please check your dataset config.")

        df_train = df_train.dropna()
        x_train, y_train = df_train["feature"], df_train["label"]

        # check if validation data is provided
        if not df_valid.empty:
            df_valid = df_valid.dropna()
            x_valid, y_valid = df_valid["feature"], df_valid["label"]
        else:
            x_valid, y_valid = None, None

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

        best_param = copy.deepcopy(self.gru_model.state_dict())
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            evals_result["train"].append(train_score)

            # evaluate on validation data if provided
            if x_valid is not None and y_valid is not None:
                val_loss, val_score = self.test_epoch(x_valid, y_valid)
                self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
                evals_result["valid"].append(val_score)

                if val_score > best_score:
                    best_score = val_score
                    stop_steps = 0
                    best_epoch = step
                    best_param = copy.deepcopy(self.gru_model.state_dict())
                else:
                    stop_steps += 1
                    if stop_steps >= self.early_stop:
                        self.logger.info("early stop")
                        break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.gru_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        # Logging
        rec = R.get_recorder()
        for k, v_l in evals_result.items():
            for i, v in enumerate(v_l):
                rec.log_metrics(step=i, **{k: v})

        if self.use_gpu:
            empty_torch_cache(self.device)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.gru_model.eval()

        # Upload features once to the accelerator (F-order handled inside) and
        # batch over the resident device tensor, mirroring the MLP predict path.
        data = to_infer_device(x_test, self.device, use_fp16=self.use_fp16, cache=self._infer_cache)

        preds = []
        with torch.no_grad():
            for begin in range(0, len(data), self.pred_batch_size):
                x = data[begin : begin + self.pred_batch_size]
                preds.append(self.gru_model(x).detach())
        preds = torch.cat(preds).cpu().numpy()

        return pd.Series(preds.reshape(-1), index=index)


class GRUModel(nn.Module):
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
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out = rnn_forward(self.rnn, x)
        return self.fc_out(out[:, -1, :]).squeeze()
