# Vision Transformer (ViT) for EuroSAT Classification

This repository contains the implementation of a Vision Transformer (ViT) model for classifying satellite images in the EuroSAT dataset. The project demonstrates the power of transformer-based models in computer vision, particularly in handling high-dimensional data like satellite imagery.

## Overview
This project demonstrates the use of the Vision Transformer (ViT) model for classifying land use types from satellite images in the EuroSAT dataset. The ViT model, known for its innovative approach to image classification by treating images as sequences of patches, has been employed here to achieve high accuracy in classifying different land use types.

## Dataset
The [EuroSAT dataset](https://github.com/phelber/eurosat) consists of 27,000 labeled images of ten different classes, representing various land use types across Europe. Each image is a 13-channel Sentinel-2 satellite image, but for this project, the dataset was processed to use the RGB channels.

### Class Labels:
- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

## Model Architecture
The Vision Transformer (ViT) model treats an image as a sequence of patches, similar to tokens in Natural Language Processing (NLP). Each patch is linearly embedded and passed through transformer layers. This approach allows the model to capture complex spatial dependencies more effectively than traditional CNNs. A tiny model of ViT was used.

### Key Features:
- **Patch Size:** 16x16
- **Transformer Layers:** 12
- **Hidden Size:** 768
- **Attention Heads**: 12

## Training Process
The model was trained for 20 epochs with a batch size of 32, using the Adam optimizer and a learning rate of 3e-4. 

### Training Metrics:
- **Initial Training Accuracy:** 46.5%
- **Final Training Accuracy:** 95%
- **Final Validation Accuracy:** 92.22%
- **Test Accuracy:** 92.67%

## Results
The ViT model achieved a test accuracy of 92.67% on the EuroSAT dataset, demonstrating its effectiveness in land use classification tasks.

## Usage
To run this project locally, follow the steps below:

1. **Clone the repository:**
   ```
   git clone https://github.com/abdulvahapmutlu/vit-eurosat-classification.git
   cd vit-eurosat-classification
   ```

2. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

3. **Prepare the dataset:**
   - Download the EuroSAT dataset from [here](https://github.com/phelber/eurosat).
   - Place the dataset folder in the code.

4. **Train the model:**
   ```
   python train.py
   ```

5. **Evaluate the model:**
   ```
   python evaluate.py
   ```

## Acknowledgments
- **EuroSAT Dataset:** P. Helber, et al. "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2019.
- **Vision Transformer:** A. Dosovitskiy, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," *International Conference on Learning Representations (ICLR)*, 2021.

---

### Scripts Used

- **train.py:** Contains the training loop and model architecture.
- **evaluate.py:** Evaluates the trained model on the test set and generates classification reports.
- **dataset.py:** Handles the loading and preprocessing of the EuroSAT dataset.
- **model.py:** Defines the Vision Transformer (ViT) architecture.

### License

This project is licensed under the MIT License.
