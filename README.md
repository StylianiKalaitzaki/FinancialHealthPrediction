# Bankruptcy Prediction Project

This project aims to predict bankruptcy for companies using machine learning techniques. The dataset used in this project contains various financial attributes of companies, and the goal is to classify weather a company is likely to go bankrupt based on these attributes.

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.x
- Libraries: pandas, scikit-learn, numpy, keras, tensorflow, matplotlib

## Usage

1. Ensure you have the dataset file named `Dataset2Use_Assignment1.xlsx` placed in the project directory.

2. Run the Python script `bankruptcy_prediction.py`.

3. The script will perform the following steps:
   - Read data from the Excel file.
   - Check for missing values in the dataset.
   - Perform exploratory data analysis, including plotting distributions.
   - Normalize the data.
   - Apply Stratified K-Fold cross-validation with 4 folds.
   - Train and evaluate various machine learning models, including Linear Discriminant Analysis, Logistic Regression, Decision Trees, Random Forests, k-Nearest Neighbors, Naïve Bayes, Support Vector Machines, and a Neural Network.
   - Calculate evaluation metrics such as accuracy, precision, recall, F1 score, and ROC-AUC score.
   - Store the results in a CSV file named `balancedDataOutcomes.csv`.

