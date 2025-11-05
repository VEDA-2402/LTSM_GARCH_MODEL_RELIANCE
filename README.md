# LTSM_GARCH_MODEL_RELIANCE
Stock Price Prediction and Volatility Forecasting

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)
![yfinance](https://img.shields.io/badge/yfinance-Data%20Extraction-yellowgreen)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-blue?logo=numpy)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-darkgreen?logo=matplotlib)
![arch](https://img.shields.io/badge/arch-GARCH%20Modeling-lightgrey)


## Project Summary
This project focuses on forecasting stock prices and estimating market volatility using a combination of deep learning and econometric modeling techniques. By leveraging historical price data, the project builds a sequential price prediction model using PyTorch-based LSTM and complements it with volatility forecasting using a GARCH model. This integrated approach provides insights into both expected price movements and associated market risk.

## Project Objective
The main objective is to develop and evaluate predictive models that can accurately forecast future stock prices and market volatility. This dual modeling aims to assist investors and analysts in making informed decisions by understanding expected price trends alongside risk patterns.

## Executive Summary of Findings
The LSTM-based model demonstrated a strong ability to capture temporal dependencies in stock prices, producing predictions closely aligned with actual market movements. Concurrently, the GARCH model effectively identified periods of heightened volatility, reflecting the clustering behavior characteristic of financial returns. Together, these models form a comprehensive tool capable of providing actionable insights into price dynamics and risk exposure.

## Data Description
This project uses historical stock price data sourced from Yahoo Finance via the `yfinance` Python package. The dataset spans from January 2008 to the present, focusing on daily closing prices of Reliance Industries Limited.

## Data Preprocessing and Modification
Raw price data was cleaned and transformed to prepare sequential inputs for the LSTM model. Specifically, closing prices were normalized using MinMax scaling to aid model convergence. Sequences of 60 days were created as features, with the subsequent day’s price as the label. For volatility analysis, daily returns were calculated and used as inputs to the GARCH model, ensuring appropriate handling of financial time series characteristics.

## Project Components
- **Preprocessing:** Handling raw data, normalization, and sequence generation.  
- **Price Prediction Model:** Deep learning LSTM network implemented in PyTorch for future price forecasting.  
- **Volatility Estimation:** GARCH(1,1) econometric modeling of returns to forecast market volatility.  
- **Evaluation & Visualization:** Metrics calculation, error visualization, and volatility plotting to assess model performance.

## Project Structure
Preprocessing.py: Loads, cleans, scales, and sequences the raw price data for modeling.

LTSM_Training.py: Defines, trains, and evaluates the PyTorch LSTM model for stock price prediction.

GARCH_Model.py: Computes returns, fits the GARCH(1,1) model, and visualizes market volatility.

LTSM_Model_Prediction.jpg: Visualization of actual vs. predicted stock prices.
GARCH_Volatility_Graph.jpg: Visualization of GARCH model’s estimated volatility over time.

## Results
-The LSTM model trained in LTSM_Training.py achieved close alignment between actual and predicted stock prices on the test set, as illustrated in the "Stock Price Prediction - Actual vs Predicted" plot. This demonstrates the model’s ability to track trends and turning points in Reliance’s stock price movements with minimal lag.​

-The GARCH(1,1) volatility model, implemented in GARCH_Model.py, successfully captured time-varying volatility and detected pronounced volatility spikes during significant market events (e.g., 2008–2009 financial crisis and the 2020 pandemic shock). The conditional volatility plot clearly shows the persistence and clustering of volatility in the time series.​

## Insights and Recommendations
-Integrating LSTM-based price forecasting with GARCH-derived volatility estimates provides a holistic view of both expected returns and risk, making this framework valuable for decision-making in risk management or active trading.

-The model’s accurate prediction of price direction suggests potential use in short-term trading and portfolio allocation, while the volatility analysis enables better timing and hedging adjustments during turbulent periods.

-For further improvements, incorporate external factors (e.g., technical indicators, volume, or macroeconomic data) and compare with alternative predictive architectures such as GRU or hybrid models.

## Limitations

- The models rely solely on historical prices and returns, excluding other market factors such as macroeconomic indicators or geopolitical events that can impact prices and volatility.
- LSTM performance is sensitive to hyperparameters and requires sufficient data length; noisy market data can reduce prediction accuracy.
- GARCH models assume conditional normality which may not fully capture extreme market shocks or fat-tailed return distributions.
- The project focuses on a single stock (Reliance Industries); generalizability to other stocks or asset classes requires further validation.



  
