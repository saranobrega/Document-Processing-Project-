import os
import pandas as pd
import json
import torch
import torch.utils.data
import torchvision
from torch.utils import data
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import RPNHead, MultiScaleRoIAlign, TwoMLPHead, FastRCNNPredictor
from pathlib import Path, PosixPath
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import wandb
import structlog
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor
import pprint

from helper.docsdataset import DocsDataset
from helper.docsdataloader import DocsDataLoader
from helper.docstrainer import DocsTrainer

logger = structlog.getLogger(__name__)


def get_device():
    device_cnt = 1
    print("")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_cnt = torch.cuda.device_count()
        logger.info(f"Number of GPUs available: {device_cnt}")
        logger.info(f"This GPU will be used: {torch.cuda.get_device_name(0)}")
    else:
        logger.info(f"No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    wandb.config.update({"GPU": {"type": torch.cuda.get_device_name(0), "count": device_cnt}})

    return device, device_cnt


def check_coco_data(root_dir, train_coco_json, val_coco_json, test_coco_json):
    logger.info("Checking coco data...")

    try:
        with open(os.path.join(root_dir, train_coco_json), "r") as file:
            train_coco_annotation = json.load(file)
        with open(os.path.join(root_dir, val_coco_json), "r") as file:
            val_coco_annotation = json.load(file)
        with open(os.path.join(root_dir, test_coco_json), "r") as file:
            test_coco_annotation = json.load(file)

        sizes = []
        image_ids_set = set()
        ann = [train_coco_annotation, val_coco_annotation, test_coco_annotation]
        sets = ["Train", "Val", "Test"]
        for (coco_annotation, set_name) in zip(ann, sets):
            for annotation in coco_annotation["annotations"]:
                image_ids_set.add(annotation["image_id"])
            logger.info(f"{set_name} set size: {len(image_ids_set)}")
            sizes.append(len(image_ids_set))
            image_ids_set.clear()

        logger.info("Coco data loaded!")
        return sizes

    except Exception as err:
        logger.error(f"Error loading coco data: {err}")


def log_train_info(config):
    logger.info("Using version of torch: {0}".format(torch.__version__))
    logger.info("Using version of torchvision: {0}".format(torchvision.__version__))
    config_dict_str = pprint.pformat(config)
    logger.info(config_dict_str)


def main():

    # load model_config
    try:
        with open("model_config.yml") as f:
            config = yaml.safe_load(f)
    except Exception as err:
        logger.error(f"Error trying loading model_config: {err}")

    # get dataset sizes: size of training data is needed to calculate steps_per_epoch
    dataset_sizes = check_coco_data(
        config["root_dir"], config["datasets"]["train"], config["datasets"]["val"], config["datasets"]["test"]
    )

    # wandb initialization
    wandb.init(
        project=config["wandb_project_name"], config=config, group=config["wandb_group_name"]
    )  # all runs for the experiment in one group
    wandb_logger = WandbLogger(project=config["wandb_project_name"], group=config["wandb_group_name"])
    logger = structlog.getLogger(__name__)

    # add dataset sizes to wandb.config
    wandb.config.update(
        {
            "subset": {
                "Train set size": dataset_sizes[0],
                "Val set size": dataset_sizes[1],
                "Test set size": dataset_sizes[2],
            }
        }
    )

    # create dataloaders
    data_module = DocsDataLoader(
        root_dir=config["root_dir"],
        coco_train_annotation_json=config["datasets"]["train"],
        coco_val_annotation_json=config["datasets"]["val"],
        coco_test_annotation_json=config["datasets"]["test"],
        batch_size=config["train_params"]["batch_size"],
        num_workers=config["train_params"]["num_workers"],
    )

    # create the model
    model = DocsTrainer(config)
    # log training params
    log_train_info(config)

    # get device (cpu or gpu)
    device, device_cnt = get_device()
    acc = "gpu" if device.type == "cuda" else "cpu"
    device_indices = list(range(0, device_cnt))
    logger.info("accelerator: " + acc)
    logger.info("devices: {0}".format(device_indices))

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[TQDMProgressBar(refresh_rate=2), lr_monitor],
        accelerator=acc,
        devices=device_indices,
        max_epochs=config["train_params"]["epochs"],
        strategy=config["trainer"]["strategy"],
    )

    # start training
    logger.info("Starting training...")
    trainer.fit(model, data_module)

    # close wandb
    wandb.finish()


if __name__ == "__main__":
    main()
