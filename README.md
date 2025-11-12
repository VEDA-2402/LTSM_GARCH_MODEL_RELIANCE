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

## Executive Summary

Financial markets exhibit complex temporal patterns and volatility clustering that traditional models struggle to capture. This project addresses the challenge of simultaneously predicting stock prices and quantifying market risk for Reliance Industries Limited using 15+ years of historical data (2008-2024). By integrating a **PyTorch-based bidirectional LSTM neural network** for price forecasting with a **GARCH(1,1) econometric model** for volatility estimation, the system achieved **96.12% explained variance (R²)** and **87.5% directional accuracy** in price predictions, while successfully detecting major volatility spikes during the 2008 financial crisis and 2020 pandemic. The dual-model approach provides actionable insights for risk-adjusted trading decisions, with RMSE of 45.23 and MAPE under 2.5%, demonstrating strong capability for both trend prediction and risk assessment in dynamic market conditions.

---

## Problem Statement

**Challenge**: Stock markets exhibit non-linear temporal dependencies and time-varying volatility that are difficult to predict using traditional linear models.

**Objective**: Develop an integrated forecasting system that can:
- Accurately predict future stock prices using historical patterns
- Quantify market risk through volatility forecasting
- Provide actionable insights for trading and risk management decisions

**Target**: Reliance Industries Limited (2008-2024 daily closing prices)

---

## Technical Stack & Methodology

### Technologies Used
- **Deep Learning**: PyTorch 2.0+ (LSTM implementation)
- **Econometrics**: ARCH library (GARCH modeling)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib
- **Data Source**: Yahoo Finance (yfinance API)


**Hyperparameters**: 30 epochs, batch size 64, Adam optimizer (lr=0.001), MSE loss

### GARCH Model
- **Type**: GARCH(1,1) with normal distribution
- **Input**: Log returns of daily closing prices
- **Output**: Conditional volatility forecasts

---

## Analysis & Visualizations

### Price Prediction Results

![Stock Price Prediction](outputs/LTSM_Model_Prediction.jpg)
*Figure 1: LSTM model predictions vs actual prices on test set*

![Prediction Analysis](outputs/stock_prediction_analysis.png)
*Figure 2: Comprehensive 4-panel analysis - Actual vs Predicted, Scatter Plot, Residuals, Error Distribution*

### Volatility Analysis

![GARCH Volatility](outputs/GARCH_Volatility_Graph.jpg)
*Figure 3: GARCH-estimated conditional volatility showing clustering during crisis periods*

![Error Analysis](outputs/error_over_time.png)
*Figure 4: Temporal distribution of prediction errors*


---

## Results

### Quantitative Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.9612 | Model explains 96% of price variance |
| **RMSE** | 45.23 | Average prediction error in price units |
| **MAE** | 32.18 | Mean absolute deviation |
| **MAPE** | 2.34% | Average percentage error |
| **Directional Accuracy** | 87.5% | Correct trend prediction rate |
| **Residual Mean** | -0.18 | Near-zero bias (unbiased predictions) |

### Key Findings

✅ **Strong Predictive Power**: The LSTM model captured complex non-linear patterns with high accuracy (R² = 0.9612)

✅ **Robust Trend Detection**: 87.5% directional accuracy enables reliable short-term trading signals

✅ **Volatility Clustering Captured**: GARCH model successfully identified:
- 2008-2009 financial crisis volatility spike
- 2020 COVID-19 pandemic market turbulence
- Periods of low volatility during stable market conditions

✅ **Low Prediction Error**: MAPE of 2.34% demonstrates practical applicability for real-world trading

✅ **Unbiased Predictions**: Residual analysis shows random distribution around zero (mean = -0.18)

### Business Impact
- **Risk Management**: Combined models enable position sizing based on predicted volatility
- **Trading Strategy**: Directional accuracy supports tactical entry/exit decisions
- **Portfolio Optimization**: Volatility forecasts improve risk-adjusted return calculations

---

## Limitations

⚠️ **Single Asset Focus**: Model trained specifically on Reliance Industries; generalization to other stocks requires retraining and validation

⚠️ **Historical Data Dependency**: Performance relies on quality and availability of past price data; limited predictive power during unprecedented market events

⚠️ **Excluded Factors**: Model doesn't incorporate:
- Macroeconomic indicators (GDP, inflation, interest rates)
- Company fundamentals (earnings, P/E ratios)
- News sentiment or geopolitical events
- Trading volume or order book data

⚠️ **GARCH Assumptions**: Conditional normality may underestimate extreme tail risks (black swan events)

⚠️ **Overfitting Risk**: Deep learning models sensitive to hyperparameters; requires regular retraining with new data

⚠️ **Computational Requirements**: Real-time deployment needs optimization for latency reduction

---





