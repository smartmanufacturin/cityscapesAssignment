## 🧠 UNet-based Semantic Segmentation on Cityscapes Dataset
<div align="center">
    <img src="https://img.shields.io/badge/PyTorch-1.13%2B-red" />
    <img src="https://img.shields.io/badge/PyTorchLightning-2.0-blue" />
    <img src="https://img.shields.io/badge/Albumentations-Transforms-success" /> 
    <img src="https://img.shields.io/badge/Cityscapes-Dataset-orange" /> 
</div>

## 📚 Project Overview

This project implements semantic segmentation using the UNet architecture with a ResNet-34 encoder, trained on the Cityscapes dataset.
It uses modern tools like PyTorch Lightning, segmentation_models_pytorch, and Albumentations for faster development and reproducibility.

✨ Key Features
🔥 UNet architecture with a ResNet-34 backbone

🚀 Fast training with PyTorch Lightning

🏆 EarlyStopping and ModelCheckpoint callbacks

🎯 Dice Loss optimization and IoU (Jaccard Index) evaluation

📈 Real-time Data Augmentation using Albumentations

🎨 Colorful visualization of segmentation masks

```
## 📂 Project Structure

VJProject/
├── src/
│   └── modeltraining.py
├── data_processing/
│   └── dataprocessing.py
├── utils/
│   └── utils.py
├── checkpoints/  # For saving model checkpoints
├── outputs   # Output visualization
    └── InputImageAndSegmentationMaskGrey.png # Input Image along with Segmentation Mask Grey
    └── InputImageAndSegmentationMaskColor.png # Input Image along with Segmentation Mask Color
    └── TestResult.png
    └── TestImagePrediction.png
    
├── requirements.txt  # Recommended for dependency management
└── trainedModels
    new_full_model_200_EPOCHS.pth
    new_model_200EPOCHS_weights.pth
    
└── README.md     # Instructions for running the project            


```
## 🚀 How to Run in Linux Environment

1. Clone the Repository

git clone https://github.com/smartmanufacturin/cityscapesAssignment.git
cd cityscapesAssignment


2. Install Requirements

pip install -r requirements.txt

3. Run the Training Script:
Navigate to the project directory and run the main script:
python src/modeltraining.py

4. Verify Outputs:
Check the checkpoints/ folder for saved model checkpoints.
Open result.png to view the visualization of the input image, ground truth mask, and predicted mask.
Review the console output for validation metrics (Mean IoU, Mean Dice, Mean Pixel Accuracy).

## 📦 Dependencies

torch

torchvision

segmentation-models-pytorch

pytorch-lightning

albumentations

torchmetrics

Pillow

matplotlib

3. Download Cityscapes Dataset
Download the dataset from the Cityscapes Website.


4. Train the Model

python train.py
or inside Kaggle/Jupyter Notebook.

## 🛠️ Model Architecture
Encoder: ResNet-34 pretrained on ImageNet

Decoder: UNet-style upsampling

Loss Function: Dice Loss

Metric: Mean IoU (Jaccard Index)

Optimizer: AdamW

Learning Rate: 1e-3

Input Size: (256 x 512)


## 🧩 Important Functions

Function                     | Purpose

encode_segmap()              | Maps original labels to training labels

decode_segmap()              | Converts label masks back to colored masks

MyClass                      | Custom Dataset class with Albumentations transforms

OurModel                     | LightningModule for training and validation


## ⚙️ Training Parameters

Parameter         | Value

Batch Size        | 16

Workers           | 4

Epochs            | 200

Early Stopping    | patience=5



## 📊 Metrics
The following metrics were used to evaluate the performance of the segmentation model:

Intersection over Union (IoU): 0.8040

Dice Coefficient: 0.8507

Pixel Accuracy: 0.92


## 🏆 Results

✅ Successfully segmented 20 valid classes from the Cityscapes dataset.

✅ Achieved high IoU scores with colorful and detailed masks.

✅ Model trained efficiently on GPU.


## 📑 References

- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Albumentations](https://albumentations.ai/)



## 🧑‍💻 Author

Sher Muhammad  
Indian Institute of Science



