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

- create_n_samples_dataset notebook (coming!) converts annotations and categories to COCO format. 

- faster_rcnn_baseline_v0 notebook is the baseline Faster-RCNN model for this project. This means no hyperparameter tuning or data augmentation was performed in this baseline model.

## Prepare the training data

### **Create coco annotations**

Use the notebook “create_n_samples_dataset.ipynb” to convert your data to COCO annotations or use existing train/val split in coco format.

Workist’s functions are used for conversion, that can also be found in Workist’s Git repository (same ones that are used for detectron2 training).

Steps:

- Set the root directory where your JSON file is located.
- “load_data_from_disc” creates annotations from a json file. You need to do it only once and it may take a while - it saves all annotations and categories and loads them
- The function “n_samples_dataset_split” takes n samples from all annotations and performs the train/val/test split.
- Save train, validation and test JSON in page annotation format.
- The code then converts the annotations to coco annotations that use either class indices [0,1,2,3,4] (choose coco_conv for conversion) or [1,2,3,4,5] (choose coco_conv_1 for conversion) - this FasterRCNN model implementation requires 1-based indexing for categories!
- The function “save_annotations” saves splitted (train/test/val) page annotations and coco annotations in the same folder.
- Finally, the function “save_index_lists” saves indices of train, validation and test set.

## Training
The **helper** folder :

- **dataloader.py -** A PyTorch Lightning data module for loading data for a document detection model. The data is loaded using the DocumentLayoutAnalysisDataset class. Here you find the train_dataloader, val_dataloader and test_dataloader.
- **dataset.py -** Reads in a dataset of images and their annotations in COCO format, and prepares the data for input into a Faster R-CNN object detection model.
- **trainer.py** - This code defines a PyTorch Lightning module that trains a Faster R-CNN model on a custom object detection dataset. The model backbone is ResNet18 and the Faster R-CNN architecture is used for object detection.
- **scheduler.py** - This code defines a custom learning rate scheduler called "WarmupMultiStepLR" which inherits from PyTorch's built-in learning rate scheduler "LambdaLR".

The **config** folder:

- The **model_config.yml** configuration file specifies various parameters and settings for training the model.
    
    In the “model_config.yml” file you can change:
    
    - name of the root_dir directory that contains coco annotation data used for training
    - The names of the coco annotation data used for training (train/val/test set)
    - Hyperparameters
    - Category-label mapping
    - Faster RCNN model parameters
    - Trainer strategy
    - Wandb configurations
    
- The **sweep_config.yml** file defines a configuration file for a hyperparameter tuning experiment. This file can be used to sweep through the different hyperparameter values using wandb sweeps.


## Weights & Biases

### **Hyperparameter optimization (Sweeps)**

For Hyperparameter optimization use the “sweep_config.yml” configuration file. Some code lines in “train.py” in wandb initialization need to be modified:

`wandb.init(`

`config=sweep_config.yml,`

`project=config["wandb_project_name"],` 

`group=config["wandb_group_name"])`

To use this, run:

`wandb sweep sweep_config.yml`

An ID will show up on the terminal. Then run:

`wandb agent your-sweep-id`

Or follow the instructions that appear on the terminal.
