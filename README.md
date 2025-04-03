### Digit_recognizer

# Digit Recognizer using KNN

![Python](https://img.shields.io/badge/Python-3.8-blue) ![scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-orange)

This project implements a digit recognition system using the **K-Nearest Neighbors (KNN) algorithm** in Python with **scikit-learn**. The model is trained on a dataset of handwritten digits and predicts the correct digit for a given image.

## Table of Contents
1. [Dataset](#dataset)
2. [Features](#features)
3. [Installation](#installation)
4. [Implementation Details](#implementation-details)
5. [Results](#results)
6. [Future Improvements](#future-improvements)

## Dataset
The dataset used in this project is sourced from **Kaggle**, containing images of handwritten digits (0-9). Each image is **28x28 pixels**, converted into a **784-dimensional feature vector** (flattened pixel values) for training and testing.

## Features
- **Image Preprocessing:** Converts 28x28 images into a structured dataframe.
- **KNN Classification:** Utilizes scikit-learn's `KNeighborsClassifier` for digit recognition.
- 📊 **Model Evaluation:** Uses train-test split and accuracy metrics for performance assessment.

## Installation
Ensure you have the following dependencies installed:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Implementation Details
### Data Preprocessing
- Reads the dataset as a Pandas dataframe.
- Normalizes pixel values for better model performance.
- Splits data into training and testing sets.

### Model Training
- Uses **KNN classifier** with Euclidean distance metric.
- Trains on labeled handwritten digits.

### Prediction & Evaluation
- Predicts digits for test images.
- Computes **accuracy** for performance analysis.

## Results
The model achieves **high accuracy** in recognizing handwritten digits. Results can vary depending on the chosen `k` value and dataset split ratio.

| k-value | Accuracy |
|---------|----------|
| 3       | 96.67%    |
| 5       | 96.49%    |
| 7       | 96.32%    |

## Future Improvements
-  Implement **Principal Component Analysis (PCA)** for dimensionality reduction.
-  Explore **Deep Learning approaches (e.g., CNNs)** for improved accuracy.
