# Fish Species Classification with Artificial Neural Network (ANN)
Akbank- 10MillionAi - Fish Classification Project

## Project Overview
This project involves classifying fish species using an Artificial Neural Network (ANN). The dataset consists of 9000 images from 9 different classes, and the project walks through the steps of data preparation, model building, training, and evaluation.

## Table of Contents
1. [Libraries and Tools](#libraries-and-tools)
2. [Dataset Handling](#dataset-handling)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Image Processing](#image-processing)
5. [Data Preparation](#data-preparation)
6. [Model Building](#model-building)
7. [Training](#training)
8. [Results Visualization](#results-visualization)
9. [Model Evaluation](#model-evaluation)
10. [Predictions](#predictions)
11. [Classification Report](#classification-report)
12. [Confusion Matrix](#confusion-matrix)
13. [Kaggle Link](#kaggle-link)

## Libraries and Tools
- **Image Processing**: `opencv-python (cv2)`
- **Parallel Computing**: `concurrent.futures (ThreadPoolExecutor)`
- **Data Manipulation**: `pandas`, `numpy`
- **Deep Learning**: `tensorflow.keras`
- And more...

## Dataset Handling
- **Directory Identification**: Dataset directory is located, and `.png` file paths are extracted.
- **Labels Creation**: Labels are derived from folder names, excluding Ground Truth (GT) images.
- **DataFrame Creation**: A pandas DataFrame is built containing file paths and labels.
- **Shuffling**: The dataset is shuffled to randomize the order.

## Exploratory Data Analysis
- Display dataset details and class distributions.
- Visualize sample images from each class.

## Image Processing
- **`load_single_image`**: Reads, resizes, and normalizes images.
- **`load_images`**: Uses parallel processing to load images efficiently.

## Data Preparation
- Split dataset into training, validation, and test sets.
- Apply image augmentation to the training data.
- One-hot encode labels.

## Model Building
A Sequential ANN model with:
- **Flatten Layer**: Converts 3D image data into 1D array.
- **Hidden Layers**: Four fully connected layers using ReLU activation.
- **Dropout Layer**: Prevents overfitting.
- **Output Layer**: Softmax activation for multi-class classification.

Compiled using:
- **Optimizer**: Adam
- **Loss**: Categorical Cross-Entropy

## Training
- Early stopping is used to terminate training when validation loss stops improving.
- Model is trained for up to 100 epochs with early stopping enabled.

## Results Visualization
- Plots of loss and accuracy for both training and validation sets are generated.

## Model Evaluation
- **model.evaluate**: Calculates loss and accuracy on the test dataset.
- Test accuracy is printed to assess model performance on unseen data.

## Predictions
- **model.predict**: Generates class predictions for the test set.
- **np.argmax**: Converts predicted probabilities into class labels.
- Actual test labels are also converted back from one-hot encoding for comparison.

## Classification Report
Generated using `classification_report`:
- **Precision**: Percentage of correct positive predictions per class.
- **Recall**: Percentage of actual positives correctly predicted.
- **F1-score**: Harmonic mean of precision and recall.
- **Support**: Number of instances for each class in the test set.

The model achieved **98% accuracy**, with high precision, recall, and F1-scores across all classes.

## Confusion Matrix
A confusion matrix is generated to visualize misclassifications. It highlights where the model confuses certain classes.
## Kaggle Lİnk
https://www.kaggle.com/code/aliyazcanigrolu/final-aiglobalhub-fish-classficiation/notebook
