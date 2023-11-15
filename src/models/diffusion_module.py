from typing import Any, Dict, Tuple
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class DiffusionModule(LightningModule):
    def __init__(self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.net = net

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()
    
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx
    ) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, _ = batch
        loss = self.net.get_loss(x, batch_idx)
        return loss

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        #self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx)
        self.log("train/loss", loss)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        x = torch.randn((16, self.net.img_depth, self.net.in_size, self.net.in_size), device = self.device)
        sample_steps = torch.arange(self.net.t_range-1, 0, -1, device=self.device)
        for t in sample_steps:
            x = self.net.denoise_sample(x, t)
        
        x = x * 0.5 + 0.5
        self.logger.log_image(
            key = "Denoise sample", images = list(x.detach())
        )
        
        return super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx)
        self.log("val/loss", loss)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=2e-4)
        return optimizer