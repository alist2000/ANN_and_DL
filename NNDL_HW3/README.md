# Traffic Sign Detection with Faster R-CNN and SSD300

This project implements and evaluates two deep learning models for object detection—**Faster R-CNN** and **SSD300**—to recognize and classify traffic signs from the German Traffic Sign Detection Benchmark (GTSDB) dataset. The primary goal is to compare the performance of a two-stage detector (Faster R-CNN) against a single-stage detector (SSD300).


## Table of Contents
* [About The Project](#about-the-project)
* [Features](#features)
* [Model Architectures](#model-architectures)
* [Results Summary](#results-summary)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation & Setup](#installation--setup)
* [Usage](#usage)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## About The Project

This repository provides a step-by-step implementation for training and evaluating object detection models using PyTorch. The project covers the entire pipeline, from data preprocessing and augmentation to model fine-tuning, evaluation, and visualization.

The key objectives are:
* To fine-tune pre-trained **Faster R-CNN (with a ResNet-50 backbone)** and **SSD300 (with a VGG16 backbone)** on the GTSDB dataset.
* To evaluate model performance using standard object detection metrics like **Intersection over Union (IoU)** and **mean Average Precision (mAP)**.
* To analyze and compare the models' effectiveness based on different IoU thresholds and object sizes (small, medium, large).

**Built With:**
* Python 3.x
* PyTorch
* torchvision
* Matplotlib
* NumPy
* OpenCV

---

## Features

* **Data Preprocessing**: Scripts to parse annotations, categorize signs into four main classes (`prohibitory`, `danger`, `mandatory`, `other`), and split the dataset into training and testing sets.
* **Data Augmentation**: Implements transformations like normalization and random horizontal flipping to improve model generalization.
* **Model Training**: Complete training loops for both Faster R-CNN and SSD300, including optimizers (AdamW/Adam) and learning rate schedulers.
* **Comprehensive Evaluation**:
    * Calculates mAP at various IoU thresholds (0.5 to 0.9).
    * Analyzes performance specifically for small, medium, and large traffic signs.
* **Visualization**: Generates plots for training loss and validation mAP, and overlays predictions and ground-truth boxes on test images for qualitative analysis.

---

## Model Architectures

### 1. Faster R-CNN with ResNet-50

A **two-stage** object detection model. It first uses a Region Proposal Network (RPN) to identify potential object locations and then uses a second network to classify these proposals and refine their bounding boxes. The ResNet-50 backbone serves as a powerful feature extractor. This architecture is known for its high accuracy. 

### 2. SSD300 with VGG16

A **single-stage** object detection model that predicts bounding boxes and class probabilities in a single pass. It uses a set of default anchor boxes at different scales and aspect ratios on multiple feature maps to detect objects of various sizes. This architecture is optimized for speed and real-time applications. 

---

## Results Summary

The evaluation showed that **Faster R-CNN significantly outperformed SSD300** across nearly all metrics on this dataset.

* **Overall Accuracy**: At an IoU threshold of 0.5, **Faster R-CNN** achieved a mean Average Precision (mAP) of **93.29%**, while **SSD300** achieved **53.55%**. 
* **Performance on Small Objects**: The difference was most dramatic for small objects, where Faster R-CNN achieved a mAP of **84.66%** compared to just **8.64%** for SSD300. This highlights the strength of the two-stage approach for detecting smaller, more challenging objects.
* **Performance vs. IoU Threshold**: Faster R-CNN maintained robust performance even at higher IoU thresholds, whereas SSD300's accuracy dropped sharply, indicating less precise bounding box predictions. 


---

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.8+
* pip
* You can install all the required libraries by running:
    ```sh
    pip install torch torchvision torchaudio matplotlib numpy opencv-python
    ```

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/traffic-sign-detection.git](https://github.com/your-username/traffic-sign-detection.git)
    cd traffic-sign-detection
    ```

2.  **Download the Dataset:**
    * Download the German Traffic Sign Detection Benchmark (GTSDB) dataset from [this link](https://benchmark.ini.rub.de/gtsdb_dataset.html). You'll need the "Full dataset" (`FullIJCNN2013.zip`).
    * Create a data directory and extract the dataset into it. The final file structure should look like this:
        ```
        traffic-sign-detection/
        ├── data/
        │   └── FullIJCNN2013/
        │       ├── 00/
        │       ├── 01/
        │       │   ...
        │       └── gt.txt
        ├── notebooks/
        │   ├── 01_prepare_dataset.ipynb
        │   ├── 02_train_faster_rcnn.ipynb
        │   └── 03_train_ssd300.ipynb
        └── README.md
        ```

---

## Usage

The project should be run in sequence using the Jupyter notebooks located in the `notebooks/` directory.

> **Important**: You must run `01_prepare_dataset.ipynb` first. [cite_start]This script processes the raw data and creates `train_files.npy` and `test_files.npy`, which are required by the other notebooks. [cite: 1408]

1.  **Prepare the Dataset:**
    * Open and run `notebooks/01_prepare_dataset.ipynb`.
    * This will analyze the dataset, create the training/test splits, and save the necessary files in the `data/FullIJCNN2013/` directory.

2.  **Train the Faster R-CNN Model:**
    * Open and run `notebooks/02_train_faster_rcnn.ipynb`.
    * This notebook will load the data, define the Faster R-CNN model, and run the training and evaluation loop. The best model weights will be saved.

3.  **Train the SSD300 Model:**
    * Open and run `notebooks/03_train_ssd300.ipynb`.
    * This notebook follows the same process for the SSD300 model.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Acknowledgments
* This project uses the [German Traffic Sign Detection Benchmark (GTSDB)](https://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset) dataset.
* The models are implemented using the excellent [PyTorch](https://pytorch.org/) library.
