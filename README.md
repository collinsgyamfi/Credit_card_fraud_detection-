# Credit Card Fraud Detection Project

## Overview
This project aims to detect fraudulent credit card transactions using machine learning. The dataset used is the Credit Card Fraud Dataset, and the model is built using LightGBM (LGB) classifier. The process involves data cleaning, handling class imbalance, feature engineering, and model evaluation.

## Table of Contents
1. [Dataset](#dataset)
2. [Libraries Used](#libraries-used)
3. [Step-by-Step Approach](#step-by-step-approach)
   - [1. Data Cleaning](#1-data-cleaning)
   - [2. Encoding Categorical Variables](#2-encoding-categorical-variables)
   - [3. Handling Imbalance](#3-handling-imbalance)
   - [4. Feature Engineering](#4-feature-engineering)
   - [5. Data Splitting](#5-data-splitting)
   - [6. Model Training](#6-model-training)
   - [7. Model Evaluation](#7-model-evaluation)
   - [8. Saving the Model](#8-saving-the-model)
   - [9. Building the Streamlit App](#9-building-the-streamlit-app)
4. [How to Run](#how-to-run)
5. [License](#license)

## Dataset
The dataset used for this project can be found at [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/dalpozz/creditcard-fraud).

## Libraries Used
- `pandas`
- `numpy`
- `lightgbm`
- `scikit-learn`
- `imbalanced-learn`
- `streamlit`
- `joblib`

## Step-by-Step Approach

### 1. Data Cleaning
- Load the dataset using `pandas`.
- Remove any irrelevant columns and handle missing values if necessary.

### 2. Encoding Categorical Variables
- Use `LabelEncoder` from `scikit-learn` to encode the following categorical variables:
  - `merchant`
  - `category`
  - `gender`

### 3. Handling Imbalance
- Check for class imbalance in the target variable.
- Use SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

### 4. Feature Engineering
- Calculate the distance between points using the Haversine distance function to add a spatial dimension to the dataset.

### 5. Data Splitting
- Split the data into training and testing sets using `train_test_split` from `scikit-learn`.

### 6. Model Training
- Train the LightGBM classifier on the training data.

### 7. Model Evaluation
- Evaluate the model using ROC curves and calculate feature importance to identify the top 10 features contributing to the model's predictions.

### 8. Saving the Model
- Save the trained model using `joblib` for future use.

### 9. Building the Streamlit App
- Create a Streamlit app to allow users to input transaction details and receive fraud detection results.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/collinsgyamfi/credit-card-fraud-detection.git
   cd credit-card-fraud-detection

