# Object detection - Document Understanding Project
Implementing Faster-RCNN model in Pytorch Lightning for an object detection project. The goal is to detect main features in document images, namely headers, tables, footers and order line items. 

Note: for privacy reasons I cannot provide the full code implementation. For more information reach me out!

## ****Pre-requisites****

- Pytorch (v14.16.1 or later)
- Pytorch Lightning (v1.8.2)
- Torchvision (v0.14.0)
- Cuda (v11.7)
- Albumentations (v1.3.0)
- Torchmetrics (v0.11.0)
- wandb (v0.13.5)

## Notebooks

- create_n_samples_dataset notebook converts annotations and categories to COCO format. 

- faster_rcnn_baseline_v0 notebook is the baseline Faster-RCNN model for this project. This means no hyperparameter tuning or data augmentation was performed in this baseline model. 
