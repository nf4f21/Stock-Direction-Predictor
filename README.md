# Stock Direction Predictor

Predicting next-day stock price movement using machine learning on historical AAPL data.

---

## Goal

Determine whether a stock's closing price will be higher or lower the following trading day, framed as a binary classification problem (1 = price increases, 0 = price decreases).

---

## Dataset

- **Source:** Yahoo Finance via the `yfinance` library
- **Ticker:** AAPL
- **Period:** January 2019 to June 2025
- **Raw features used:** Close, Volume

---

## Feature Engineering

| Feature | Description |
|---|---|
| Percentage Change | Daily return: (today - yesterday) / yesterday * 100 |
| 5-Day Moving Average | Rolling mean of closing price over 5 trading days |
| Price Momentum | 10-day return: (today - 10 days ago) / 10 days ago * 100 |
| Volume | Raw daily trading volume |

**Target:** Binary label created using a forward shift on the closing price. If tomorrow's close is greater than today's the label is 1, otherwise 0.

---

## Pipeline

1. Download historical data
2. Engineer features
3. Drop NaN rows produced by shifting and rolling operations
4. Remove the final row (no forward label available)
5. Train/test split (80/20, no shuffle to preserve time ordering)
6. Scale features using StandardScaler fitted on training data only
7. Train and evaluate models

---

## Models

### Logistic Regression
A linear classifier used as a baseline for binary classification.

- Without scaling: predicted all 1s (majority class bias)
- With scaling: accuracy ~44%, heavily biased toward predicting 0s

### Random Forest (100 estimators)
An ensemble of decision trees that handles non-linear patterns without requiring feature scaling.

- Accuracy: ~48%
- Predictions collapsed toward 0s in the latter half of the test period

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression (unscaled) | ~56% (predicts all 1s) |
| Logistic Regression (scaled) | ~44% |
| Random Forest | ~48% |

No model meaningfully outperformed a naive baseline. The confusion matrices confirmed that models were finding shortcuts related to class distribution rather than learning genuine signal from the features.

---

## Key Learning

This project demonstrated that simple models struggle to predict stock price direction due to market randomness and noise. Technical indicators derived purely from price and volume carry limited predictive signal for next-day direction. The market reflects a vast amount of information that these features do not capture.

Equally important were the engineering lessons: data leakage through improper scaling order, the misleading nature of accuracy as a sole metric, the difference between a model that appears to perform well and one that is genuinely learning, and the importance of evaluating models with a confusion matrix rather than a single number.

---

## Stack

- Python
- pandas, numpy
- yfinance
- scikit-learn
- matplotlib
