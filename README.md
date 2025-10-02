# Machine Learning Projects and Algorithms

This repository contains a collection of Python scripts and projects related to a machine learning course. The projects cover a wide range of topics, from fundamental data analysis techniques to the implementation and explanation of various machine learning algorithms.

## High-Level Summary

For business leaders and project managers, this repository serves as a practical showcase of machine learning concepts applied to real-world datasets. It demonstrates the implementation of key algorithms that can be used to derive insights from data, such as predicting health outcomes or segmenting customers. This collection can be a valuable resource for understanding the technical capabilities of a data science team and for identifying potential applications of machine learning within your organization.

## Technical Deep-Dive

For developers and data scientists, this repository offers a hands-on approach to learning and implementing machine learning models. The scripts are well-commented and provide clear explanations of the underlying algorithms. The code is written in Python and utilizes popular libraries such as NumPy, Pandas, and Scikit-learn.

### Algorithms and Concepts Covered:

*   **Principal Component Analysis (PCA)**: Dimensionality reduction technique.
*   **Hierarchical Clustering**: Agglomerative clustering for identifying data hierarchies.
*   **Linear Regression**: A foundational regression algorithm.
*   **Artificial Neural Networks (ANN)**: A multi-layer perceptron for classification.
*   **K-Means Clustering**: An iterative clustering algorithm.
*   **AdaBoost**: An ensemble learning method for boosting classification performance.
*   And many more, including regularization, cross-validation, and probability calculations.

### Datasets

The primary dataset used in these projects is the **South African Heart Disease dataset** (`saheart_1_withheader.csv`), which is used for classification tasks to predict the presence of heart disease.

## Repository Structure

The repository is organized into four main directories:

*   `01/`: Contains introductory projects, including data exploration, PCA, and hierarchical clustering on the heart disease dataset.
*   `02/`: Focuses on classification models and conditional probability calculations.
*   `03/`: Includes additional exercises and exam-related scripts.
*   `04/`: A comprehensive collection of scripts implementing various machine learning algorithms from scratch or using libraries like Scikit-learn.

## Getting Started

### Prerequisites

To run the scripts in this repository, you will need Python 3 and the following libraries:

*   NumPy
*   Pandas
*   Matplotlib
*   Scikit-learn

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Running the Scripts

Navigate to the desired project directory and run the Python scripts directly from your terminal. For example, to run the linear regression script:

```bash
cd 04/
python linear_regression.py
```