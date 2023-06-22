import numpy as np
import pandas as pd
import os
import torch
import torch.utils.data
import torchvision
from torch.utils import data
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.optim import AdamW, SGD
import pytorch_warmup as warmup
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl
import wandb
import structlog
from pytorch_lightning.loggers import WandbLogger
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN

from helper.warmup_scheduler import WarmupMultiStepLR


logger = structlog.getLogger(__name__)


def get_resnet18_backbone_model(config, pretrained):

    print("Using fasterrcnn with resnet18 backbone...")

    # resnet_fpn_backbone() sets backbone.out_channels to the correct value automatically
    backbone = resnet_fpn_backbone(config["model_params"]["backbone_name"], pretrained=pretrained)

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)), aspect_ratios=tuple([(0.5, 1.0, 2.0, 4.0, 8.0) for _ in range(5)])
    )

    model = FasterRCNN(
        backbone,
        num_classes=config["model_params"]["num_classes"],
        rpn_anchor_generator=anchor_generator,
        rpn_pre_nms_top_n_train=config["model_params"]["rpn_pre_nms_top_n_train"],
        rpn_pre_nms_top_n_test=config["model_params"]["rpn_pre_nms_top_n_test"],
        box_score_thresh=config["model_params"]["box_score_thresh"],
    )

    return model


class DocsTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.id2label = config["id2label"]
        self.lr = config["train_params"]["lr"]
        self.warmup_steps = int(config["multistepLR"]["warmup_steps"] / 2)
        self.milestones = [int(step / 2) for step in list(config["multistepLR"]["milestones"])]
        self.gamma = config["multistepLR"]["gamma"]
        self.weight_decay = config["train_params"]["weight_decay"]

        self.map = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        self.model = get_resnet18_backbone_model(config=self.config, pretrained=False)

        # log model architecture
        # logger.info(self.model)

        self.save_hyperparameters()
        wandb.watch(self.model, criterion=None, log="all")

    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        warmup_steps = self.warmup_steps
        milestones = self.milestones
        gamma = self.gamma
        optimizer = SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = WarmupMultiStepLR(optimizer, warmup_steps, milestones, gamma)

        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step"))

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)  # dict with tensors that contain loss values(cls,rpn...)
        train_loss = sum(loss for loss in loss_dict.values())

        wandb.log(loss_dict)
        wandb.log({"train_loss": train_loss})

        return {"loss": train_loss, "outputs": {k: v.detach() for k, v in loss_dict.items()}}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        preds = self.model(images, targets)

        self.map.update(preds=preds, target=targets)

        return {"preds": preds[0], "targets": targets[0]}

    def validation_epoch_end(self, outputs):

        # compute mAP
        mAPs = {
            k: np.asarray(v)
            for k, v in self.map.compute().items()
            if k in ["map", "map_50", "map_75", "map_large", "map_per_class"]
        }

        mAPs_per_class = mAPs.pop("map_per_class")

        logger.info("Epoch {}:".format(self.current_epoch))
        mAPs = {k: float(v) for k, v in mAPs.items()}
        df1 = pd.DataFrame(mAPs.items(), columns=["metric", "value"])

        mAPs_per_class_dict = dict()
        for label, value in zip(self.id2label.values(), mAPs_per_class):
            key = "mAP_{}".format(label)
            mAPs_per_class_dict[key] = float(value)

        df2 = pd.DataFrame(mAPs_per_class_dict.items(), columns=["metric", "value"])
        df = pd.concat([df1, df2]).reset_index(drop=True)
        df["metric"] = df["metric"].str[:30]
        logger.info(repr(df))

        wandb.log(mAPs)
        wandb.log(mAPs_per_class_dict)

        self.map.reset()
