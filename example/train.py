import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.datasets as vdata
import torchvision.transforms as vxform
from tqdm import tqdm

from mdlr import state, trainer
from mdlr.logging import Logger, TensorboardAdapter
from mdlr.serialize import SerializableData
from mdlr.utils import RunningMean, default_field, loop

from .model import ModelParam, ModelState, ModelStateManager


@dataclass(kw_only=True)
class TrainParam(SerializableData):
    max_epochs: int = 100
    "Maximum number of epochs to train"
    lr: float = 1e-3
    "Learning rate"
    lr_step: int = 1
    "Step size for StepLR"
    lr_gamma: float = 0.99
    "Gamme for StepLR"
    batch_size: int = 64
    "Batch size for both train and val"
    num_workers: int = (os.cpu_count() or 0) // 2
    "Number of workers (default os.cpu_count() // 2)"


@dataclass(kw_only=True)
class TrainState(SerializableData):
    optim: torch.optim.Optimizer
    sched: torch.optim.lr_scheduler.LRScheduler
    logger: Logger = default_field(Logger)
    train_loss: RunningMean = default_field(RunningMean)
    valid_loss: RunningMean = default_field(RunningMean)
    current_epoch: int = 0


class TrainStateManager(
    state.ManagedState[
        TrainParam,
        TrainState,
        ModelStateManager,
    ],
    param_t=TrainParam,
    state_t=TrainState,
):
    @classmethod
    def configure(cls, param: TrainParam, mstate: ModelStateManager) -> TrainState:
        optim = torch.optim.Adam(mstate.state.model.parameters(), lr=param.lr)
        sched = torch.optim.lr_scheduler.StepLR(optim, param.lr_step, param.lr_gamma)
        return TrainState(optim=optim, sched=sched)


class Trainer(
    trainer.Trainer[
        ModelParam,
        ModelState,
        TrainParam,
        TrainState,
        ModelStateManager,
    ]
):
    mstate_m = ModelStateManager
    tstate_m = TrainStateManager

    def get_data_loaders(self) -> tuple[tdata.DataLoader, tdata.DataLoader]:
        persisitent_workers = self.tparam.num_workers > 0
        xform = vxform.Compose([vxform.ToTensor(), vxform.Normalize(0.5, 0.5)])
        train_ds = vdata.CIFAR10("../data", train=True, download=True, transform=xform)
        valid_ds = vdata.CIFAR10("../data", train=False, download=True, transform=xform)

        shared_params = dict(
            batch_size=self.tparam.batch_size,
            pin_memory=True,
            num_workers=self.tparam.num_workers,
            persistent_workers=persisitent_workers,
        )

        train_dl = tdata.DataLoader(
            train_ds,
            shuffle=True,
            **shared_params,
        )

        valid_dl = tdata.DataLoader(
            valid_ds,
            shuffle=False,
            **shared_params,
        )

        return train_dl, valid_dl

    def get_aux(self, label: torch.Tensor) -> torch.Tensor | None:
        if self.mparam.aux_width == 0:
            return None
        label = nn.functional.one_hot(label, self.mparam.aux_width).float()
        label = label.unsqueeze(-1).unsqueeze(-1)
        return label

    def get_inputs(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, y = batch
        x = x.to(self.mparam.device)
        y = y.to(self.mparam.device)
        y = self.get_aux(y)
        return x, y

    def get_loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = self.get_inputs(batch)
        xhat: torch.Tensor = self.mstate.model(x, y)
        loss = nn.functional.mse_loss(xhat, x)
        return loss

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.tstate.optim.zero_grad(True)
        loss = self.get_loss(batch)
        loss.backward()
        self.tstate.optim.step()
        self.tstate.train_loss += loss.item()
        if self.tstate.train_loss.count == 20:
            self.tstate.logger.add_scalar(
                "loss/train", self.tstate.train_loss.flush(), inc=20
            )

    @torch.no_grad()
    def valid_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        loss = self.get_loss(batch)
        self.tstate.valid_loss += loss.item()

    @torch.no_grad()
    def sample(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        x, y = self.get_inputs(batch)
        x = x[:8]
        if y is not None:
            y = y[:8]
        xhat: torch.Tensor = self.mstate.model(x, y)
        img = torch.cat([x.unsqueeze(0), xhat.unsqueeze(0)])
        self.tstate.logger.add_image("sample", img)
        pass

    def train(self, dir: str) -> None:
        self.tstate.logger.add_adapters(
            TensorboardAdapter(dir),
        )

        self.mstate.model.to(self.mparam.device)

        train_dl, valid_dl = self.get_data_loaders()

        for epoch in range(self.tstate.current_epoch, self.tparam.max_epochs):
            self.tstate.current_epoch = epoch

            self.save(dir)

            loop(self.train_step, tqdm(train_dl, f"Train {epoch + 1}", leave=False))
            if self.tstate.train_loss.count > 0:
                self.tstate.logger.add_scalar(
                    "loss/train", self.tstate.train_loss.flush()
                )

            loop(self.valid_step, tqdm(valid_dl, f"Valid {epoch + 1}", leave=False))
            self.tstate.logger.add_scalar("loss/valid", self.tstate.valid_loss.flush())
            self.sample(next(iter(valid_dl)))

            self.tstate.sched.step()

        self.tstate.current_epoch = self.tparam.max_epochs
