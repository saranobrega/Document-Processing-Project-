import os
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
import albumentations
import torch
import torch.utils.data
import torchvision
from torch.utils import data
from torchvision import transforms


def transform_image_and_bboxes(image_arr, bboxes, class_labels, h, w):

    """
    resizes the images and bboxes to width w and height h,
    transforms images, bboxes and class_labels to tensors needed as input to FasterRCNN model
    """

    transform = albumentations.Compose(
        [albumentations.Resize(height=h, width=w, always_apply=True)],
        bbox_params=albumentations.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    transformed = transform(image=image_arr, bboxes=bboxes, class_labels=class_labels)

    # image vector elements in range [0,255] - need to be in range [0,1]
    img_tensor_transform = transforms.Compose([transforms.ToTensor()])

    transformed["image"] = img_tensor_transform(transformed["image"])

    # after resizing bboxes are coordinates are tuples - convert to lists
    for i in range(0, len(transformed["bboxes"])):
        transformed["bboxes"][i] = list(transformed["bboxes"][i])

    # convert bboxes and labels to tensors
    transformed["bboxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
    transformed["class_labels"] = torch.as_tensor(transformed["class_labels"], dtype=torch.int64)

    return transformed


class DocsDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation):
        self.width = 600
        self.height = 800
        self.root = root
        print("Creating coco dataset from {0} file:".format(annotation))
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        annotation_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(annotation_ids)
        # path for input image:
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image (type PIL)
        img = Image.open(os.path.join(self.root, path))

        # number of objects(bboxes) in the image
        num_objs = len(coco_annotation)

        # tensor containing object categories
        category_ids = []
        for i in range(len(coco_annotation)):
            category_ids.append(coco_annotation[i]["category_id"])
        class_labels = category_ids

        # Bounding boxes for objects
        # coco format bbox = [xmin, ymin, width, height]
        # pascal_voc format bbox = [xmin, ymin, xmax, ymax] -> required input format for pytorch faster rcnn
        pascal_boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            w = coco_annotation[i]["bbox"][2]
            h = coco_annotation[i]["bbox"][3]
            xmax = xmin + w
            ymax = ymin + h
            pascal_boxes.append([xmin, ymin, xmax, ymax])

        # Tensorise img_id
        img_id = torch.tensor([img_id])

        # Area of the bbox
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # resize images and bounding boxes
        image_arr = np.array(img)
        bboxes_arr = np.array(pascal_boxes)

        transformed = transform_image_and_bboxes(image_arr, bboxes_arr, class_labels, self.height, self.width)
        # print(transformed)

        img = transformed["image"]

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = transformed["bboxes"]
        my_annotation["labels"] = transformed["class_labels"]
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        my_annotation

        return img, my_annotation

    def __len__(self):
        return len(self.ids)
