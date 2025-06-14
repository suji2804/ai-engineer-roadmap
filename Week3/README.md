# Week 3 - Credit Card Fraud Detection with XGBoost

This project is part of my AI Engineer learning roadmap. The goal is to build a robust fraud detection model using tree-based ensemble methods, especially XGBoost, to handle a highly imbalanced classification problem.

## 🧾 Project Overview

- **Dataset**: Credit Card Fraud Detection ([Kaggle Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- **Objective**: Predict whether a credit card transaction is fraudulent or not
- **Challenge**: Severe class imbalance (fraud < 1%)

## 🧠 Tasks Performed

### 🔹 Part 1: Data Preprocessing & EDA
- Analyzed class imbalance and distribution of transaction amount/time
- Scaled `Amount` and `Time` features using `StandardScaler`
- Handled imbalance with stratified train-test split
- Optionally explored SMOTE for oversampling minority class

### 🔹 Part 2: Model Training
- Trained baseline model with `Logistic Regression`
- Trained ensemble models:
  - **Random Forest**
  - **XGBoost Classifier**
- Evaluated using:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC Curve
  - Feature importance plots

---

## 📊 Results
| Model             | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------------|----------|-----------|--------|----------|--------- |
| LogisticRegression |0.9884  | 0.1191    | 0.8980 | 0.2103   | 0.9433   |
| RandomForest      | 0.9994  |   0.8265  | 0.8265 |  0.8265  | 0.9131   |
| XGBoost (Best)    |0.9994   |   0.7905  | 0.8469 |  0.8177  | 0.9233   |

> Replace the table above with actual metrics after running your notebook.



