import os
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils import data
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd


from helper.docsdataset import DocsDataset


def collate_fn(batch):
    return tuple(zip(*batch))


class DocsDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: Path,
        coco_train_annotation_json: "",
        coco_val_annotation_json: "",
        coco_test_annotation_json: "",
        batch_size: int,
        num_workers: int,
        stage: int = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.coco_train_annotation_json = coco_train_annotation_json
        self.coco_val_annotation_json = coco_val_annotation_json
        self.coco_test_annotation_json = coco_test_annotation_json
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stage = stage

    def __post_init__(self):
        super().__init__()
        self.setup(self.stage)

    def setup(self, stage=None):

        self.train_ds = DocsDataset(
            root=self.root_dir, annotation=os.path.join(self.root_dir, self.coco_train_annotation_json)
        )
        self.val_ds = DocsDataset(
            root=self.root_dir, annotation=os.path.join(self.root_dir, self.coco_val_annotation_json)
        )
        self.test_ds = DocsDataset(
            root=self.root_dir, annotation=os.path.join(self.root_dir, self.coco_test_annotation_json)
        )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers
        )
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers
        )
        return test_loader
