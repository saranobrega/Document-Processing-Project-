# Object detection - Document Processing Project
Implementing the Faster-RCNN model for an object detection project. The goal is to detect main features in document transactions, namely headers, tables, footers and order line items. 


## Technologies
This project is created with:
- Python 3.10.8
- Pytorch 1.12.0
- Pytorch Lightning 1.7.7
- Pandas 1.5.1
- Numpy 1.23.4
- Seaborn 0.12.1
- Wandb 0.13.5

## Notebooks

- Convert_to_coco notebook converts annotations and categories to COCO format. 

- faster_rcnn_baseline_v0 notebook is the baseline Faster-RCNN model for this project. This means no hyperparameter tuning or data augmentation was performed in this baseline model. 
