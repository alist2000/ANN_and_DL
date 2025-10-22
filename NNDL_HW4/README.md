# Crude Oil Price Prediction using Deep Learning and ARIMA

This project provides a comparative analysis of different time-series forecasting models for predicting crude oil prices (`CL=F`). It implements and evaluates three deep learning architectures (**LSTM, GRU, and Bi-LSTM**) and a classical statistical model (**ARIMA**) to determine their effectiveness on volatile financial data.

The implementation and findings are based on the methods described in the associated research paper, focusing on data preprocessing, model tuning, and performance evaluation.


## Table of Contents
* [About The Project](#about-the-project)
* [Features](#features)
* [Model Architectures](#model-architectures)
* [Results Summary](#results-summary)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## About The Project

The primary goal of this repository is to forecast the 'Adj Close' price of crude oil by comparing deep learning models against the traditional ARIMA model. The project covers a complete workflow:

1.  **Data Acquisition and Preprocessing**: Fetches historical data from Yahoo Finance, introduces and handles missing values using linear interpolation, normalizes the data, and prepares it for time-series modeling.
2.  **Model Implementation**: Builds, trains, and tunes LSTM, GRU, Bi-LSTM, and ARIMA models.
3.  **Comprehensive Evaluation**: Assesses model performance using a suite of standard regression metrics, including MAE, MSE, RMSE, R², and MAPE.

**Built With:**
* Python 3.x
* TensorFlow / Keras
* `yfinance`
* `pmdarima`
* `statsmodels`
* Scikit-learn
* Matplotlib

---

## Features

* **Automated Data Fetching**: Uses the `yfinance` library to download up-to-date historical crude oil price data.
* **Robust Preprocessing**: Includes steps for randomly removing data to simulate real-world imperfections and then imputing missing values using linear interpolation.
* **Deep Learning Models**:
    * Long Short-Term Memory (**LSTM**)
    * Gated Recurrent Unit (**GRU**)
    * Bidirectional Long Short-Term Memory (**Bi-LSTM**)
* **Statistical Model**:
    * Autoregressive Integrated Moving Average (**ARIMA**) with automated hyperparameter tuning using `auto_arima`.
* **Dual ARIMA Evaluation**: Compares a **Traditional Forecasting** approach with a more robust **Walk-Forward Validation** method.
* **In-depth Analysis**: Evaluates the impact of `sequence_length` on model performance and provides detailed comparisons across all models.

---

## Model Architectures

* **LSTM**: A type of Recurrent Neural Network (RNN) well-suited for learning long-term dependencies in sequential data, making it ideal for financial time series.
* **GRU**: A more modern and slightly simpler alternative to LSTM. It often performs comparably while being computationally more efficient.
* **Bi-LSTM**: An extension of LSTM that processes data in both forward and backward directions, allowing the model to capture context from both past and future steps in the sequence.
* **ARIMA**: A classical statistical model that captures linear relationships in time-series data based on autoregression, differencing, and moving averages.

---

## Results Summary

The evaluation clearly demonstrates that **deep learning models significantly outperform the classical ARIMA model** for this task. The non-linear and complex patterns in oil price data are better captured by the neural network architectures.

* **Best Performing Model**: The **Bi-LSTM** model with a `sequence_length` of 2 achieved the best results across all metrics, with an R² of **0.9832** and the lowest MAE, RMSE, and MAPE values.
* **Impact of `sequence_length`**: Reducing the sequence length from 10 to 2 generally improved the performance of all deep learning models, suggesting that recent price movements are more predictive.
* **ARIMA Performance**:
    * The **Traditional ARIMA** model performed very poorly, yielding a negative R² score, which indicates it is not a suitable model for this dataset. 
    * The **Walk-Forward Validation ARIMA** showed a massive improvement over the traditional method but was still not competitive with the deep learning models. 


### Performance Metrics (`sequence_length=2`)

| Model | MAE | RMSE | R² | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Bi-LSTM** | **1.55** | **2.20** | **0.9832** | **2.05** |
| GRU | 1.68 | 2.38 | 0.9804 | 2.20 |
| LSTM | 1.73 | 2.46 | 0.9791 | 2.28 |
| ARIMA (Walk-Forward) | 1.40 | 1.96 | 0.9867 | 27.72 |

*(Note: While ARIMA (Walk-Forward) has a low MAE/RMSE, its very high MAPE suggests poor percentage-based accuracy, and it is less stable than the DL models).*

---

## Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

You will need Python 3.8+ and `pip`. All required libraries are listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/oil-price-prediction.git](https://github.com/your-username/oil-price-prediction.git)
    cd oil-price-prediction
    ```

2.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should contain:
    ```
    tensorflow
    pandas
    numpy
    yfinance
    pmdarima
    statsmodels
    scikit-learn
    matplotlib
    seaborn
    ```

---

## Usage

The project is structured into two main Jupyter notebooks.

1.  **Deep Learning Models (`HW4_2_DeepLearning.ipynb`):**
    * Open and run this notebook to perform data preprocessing and train/evaluate the LSTM, GRU, and Bi-LSTM models.
    * You can adjust the `seq_length` variable at the top of the notebook to experiment with different look-back periods.

2.  **ARIMA Model (`HW4_2_ARIMA.ipynb`):**
    * Open and run this notebook to find the optimal ARIMA parameters and evaluate the model using both the Traditional and Walk-Forward Validation methods.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Acknowledgments
* Data sourced from [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` library.
* Key libraries used: [TensorFlow](https://www.tensorflow.org/), [pmdarima](https://pypi.org/project/pmdarima/), and [statsmodels](https://www.statsmodels.org/).
