# Wind Power Forecasting using Transformer, Autoencoder, and Huber Loss

This project implements and evaluates advanced deep learning models for time-series forecasting of offshore wind power. It explores the use of a Transformer-based architecture, denoising with an Autoencoder, and the robustness of the Huber loss function, based on the methodologies presented in the associated research paper.



## Table of Contents
* [About The Project](#about-the-project)
* [Key Features](#key-features)
* [Methodology Overview](#methodology-overview)
* [Results Summary](#results-summary)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## About The Project

Forecasting wind power is challenging due to its high volatility, presence of outliers, and non-linear patterns. Traditional statistical models like ARIMA often fail to capture these complexities. This project tackles these challenges by:

1.  **Denoising Data:** Using a **TensorFlow-based Autoencoder** to reconstruct the raw time-series data, effectively removing noise and outliers.
2.  **Advanced Forecasting:** Implementing and comparing three **PyTorch-based** deep learning models: a **Transformer**, a standard **RNN**, and a simple **MLP**.
3.  **Robust Training:** Evaluating the models with two different loss functions: the standard **Mean Squared Error (MSE)** and the outlier-resistant **Huber Loss**.
4.  **Hyperparameter Optimization:** (Bonus) Utilizing the **Slime Mould Algorithm (SMA)** to find the optimal hyperparameters for the Transformer model.

The project explores both **single-step** and **multi-step** forecasting scenarios.

**Built With:**
* Python 3.x
* PyTorch
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib

---

## Key Features

* **Outlier Detection**: Implements a rolling window algorithm to identify anomalous data points in the time series.
* **Autoencoder for Denoising**: A deep Autoencoder is trained to clean the input signal before feeding it to the forecasting models.
* **Transformer Model**: A state-of-the-art Transformer architecture with positional encoding and multi-head self-attention for capturing long-range dependencies.
* **Huber Loss Function**: A custom implementation of the Huber loss to reduce the model's sensitivity to outliers during training.
* **Slime Mould Algorithm (SMA)**: A nature-inspired metaheuristic algorithm used to automate the search for optimal model hyperparameters.
* **Single vs. Multi-Step Forecasting**: The Transformer model is evaluated on its ability to predict a single future time step as well as multiple future steps (4, 8, and 16 steps ahead).

---

## Methodology Overview

The project follows a systematic pipeline to ensure robust and reproducible results:

1.  **Data Loading**: The first 1440 data points from the `WT5` dataset are used.
2.  **Outlier Detection**: Anomalies in the raw signal are identified.
3.  **Denoising**: The data is partitioned into sliding windows (size 144) and reconstructed using a trained Autoencoder.
4.  **Data Splitting**: The cleaned data is split into an 80% training set and a 20% testing set.
5.  **Normalization**: The training data is fitted with a `MinMaxScaler`, which is then used to transform both the training and testing sets.
6.  **Model Training & Evaluation**: The Transformer, RNN, and MLP models are trained and evaluated using both MSE and Huber loss functions.

---

## Results Summary

### Single-Step Forecasting

We conducted two sets of experiments for single-step prediction.

#### 1. Using All 8 Features (As per Paper)
When using all 8 input features, the **RNN model trained with MSE loss performed the best**, showing the lowest error across all metrics. This indicates that for this dataset, a simpler recurrent architecture was more effective than the more complex Transformer.

| Model | Loss | MAE | RMSE | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| MLP | Huber | **2.62** | 3.66 | **6.50** |
| **RNN** | **MSE** | 2.88 | **3.54** | 6.99 |
| Transformer| MSE | 5.54 | 7.16 | 13.38 |

#### 2. Using Only Wind Power as a Feature (Additional Analysis)
In a supplementary experiment, we trained the models using only the wind power time series as input (1 feature instead of 8). This approach yielded **dramatically better results**, outperforming both our previous results and those reported in the paper.

The **RNN model with Huber loss** was the clear winner, achieving an RMSE of just **0.48**.

| Model | Loss | MAE | RMSE | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **RNN** | **Huber** | **0.43** | **0.48** | **0.66** |
| RNN | MSE | 0.64 | 0.71 | 0.94 |
| Transformer| Huber | 1.03 | 1.18 | 1.48 |

This suggests that the additional features (like humidity, air density, etc.) may have introduced more noise than useful information, and a focused, autoregressive approach is more effective.

### Multi-Step Forecasting (Transformer Model)

For multi-step forecasting, the model's accuracy understandably decreased as the prediction horizon increased. While our model's MAE and RMSE were significantly better than the paper's, our MAPE was higher, suggesting our model struggled more with relative errors.

| Horizon | MAE | RMSE | MAPE (%) |
| :--- | :--- | :--- | :--- |
| **t+4** (40 mins) | 8.71 | 10.40 | 11.48 |
| **t+8** (80 mins) | 11.72 | 14.25 | 15.72 |
| **t+16** (160 mins) | 19.52 | 22.79 | 27.64 |

---

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.8+
* pip

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/wind-power-forecasting.git](https://github.com/your-username/wind-power-forecasting.git)
    cd wind-power-forecasting
    ```

2.  **Install the required packages:**
    ```sh
    pip install torch tensorflow numpy pandas matplotlib scikit-learn
    ```

3.  **Download the Dataset:**
    * Download the dataset file (`WT5.csv`) mentioned in the report.
    * Place it in the root directory of the project.

---

## Usage

The project is organized into Jupyter notebooks. It is recommended to run them in the following order:

1.  **`HW5_1_Data_Preprocessing_and_Single_Step.ipynb`**:
    * This notebook contains the code for outlier detection, training the denoising Autoencoder, and training/evaluating the MLP, RNN, and Transformer models for single-step forecasting.

2.  **`HW5_1_Multi_Step_Transformer.ipynb`**:
    * This notebook focuses on implementing and evaluating the Transformer model for multi-step forecasting (t+4, t+8, t+16).

3.  **`HW5_1_Slime_Mould_Optimization.ipynb`** (Optional/Bonus):
    * This notebook demonstrates how to use the Slime Mould Algorithm to find optimal hyperparameters for the Transformer model.

---

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

---

## Acknowledgments
* This work is an implementation based on the methods described in the referenced academic paper.
* The project utilizes the `WT5` offshore wind power dataset.
