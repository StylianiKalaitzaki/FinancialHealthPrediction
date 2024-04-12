import sys
import pandas as pd
import sklearn
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,precision_score, recall_score,f1_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Function to read data from an Excel file
def read_excel_data(file_name, sheet_name):
    try:
        sheet_values = pd.read_excel(file_name, sheet_name)
        print(' .. successful parsing of file:', file_name)
        return sheet_values
    except FileNotFoundError:
        print('File not found:', file_name)
        return None

# Function to create a subplot with scatter plots for min, max, and mean values
def create_subplot(ax, min_values,max_values,avrg_values, index_names,title):
    ax.scatter(index_names, min_values.values, label='Min Value', color='blue', marker='o', alpha=0.5, s=80)
    ax.scatter(index_names, max_values.values, label='Max Value', color='red', marker='+', s=90)
    ax.scatter(index_names, avrg_values.values, label='Mean Value', color='green', marker='x', s=85)
    ax.set_ylim(y_min - 100, y_max + 100)
    ax.set_xticks(index_names)
    ax.set_xticklabels(index_names, rotation=45, ha='right')
    ax.set_xlabel('Indexes')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()

# Function to check for missing values in the dataset
def check_missing_values(data):
  #Checks for 'None' values in data
    missing_values = data.isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing records.")
    else:
        print("No missing records found.")

# Function to display confusion matrix
def display_confusion_matrix(y_true,y_pred,model_name,set_type):
  # Calculate the confusion matrix
  cm = confusion_matrix(y_true, y_pred)

  # Display the confusion matrix using ConfusionMatrixDisplay
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Healthy', 'Bankrupt'])
  disp.plot(cmap=plt.cm.Blues)

  plt.title(f'{model_name} Confusion Matrix {set_type} Set')
  plt.show()

  # Return the confusion matrix as well
  return cm

# Function to calculate various metrics and print the scores
def calculate_metrics(y_train,y_test,y_pred_train,y_pred_test):
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    pre_train = precision_score(y_train, y_pred_train, zero_division=0) #when there are no predicted samples for a class#!!!check about the average arg
    pre_test = precision_score(y_test, y_pred_test, zero_division=0)
    rec_train = recall_score(y_train, y_pred_train)
    rec_test = recall_score(y_test, y_pred_test)
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)
    roc_auc_train = roc_auc_score(y_train, y_pred_train)
    roc_auc_test = roc_auc_score(y_test, y_pred_test)

    #[8.c] print the scores
    print('Accuracy scores are: train: {:.2f}'.format(acc_train), 'and test: {:.2f}.'.format(acc_test))
    print('Precision scores are: train: {:.2f}'.format(pre_train), 'and test: {:.2f}.'.format(pre_test))
    print('Recall scores are: train: {:.2f}'.format(rec_train), 'and test: {:.2f}.'.format(rec_test))
    print('F1 scores are: train: {:.2f}'.format(f1_train), 'and test: {:.2f}.'.format(f1_test))
    print('AUC ROC scores are: train: {:.2f}'.format(roc_auc_train), 'and test: {:.2f}.'.format(roc_auc_test))

    return roc_auc_train,roc_auc_test

# Function to load information into a DataFrame
def load_info(df,model_name,set_type, balance, X_train,bankrupt_companies,cm, auc_roc):
  #cm : Confusion Matrix
  info = {
        'Classifier Name': model_name,
        'Set Type': set_type,
        'Balance': balance,
        'N training samples': len(X_train),
        'N non-healthy companies in training sample': bankrupt_companies,
        'TP': cm[1,1],
        'TN': cm[0,0],
        'FP': cm[0,1],
        'FN': cm[1,0],
        'ROC-AUC': auc_roc
    }

  df = df.append(info, ignore_index=True)
  return df

# Function to train and evaluate a classifier
def train_and_evaluate_classifier(df,ml_model, X_train, y_train, X_test, y_test, ml_model_name, balance,bankrupt_companies):
  # Train the model
  if ml_model_name != "Neural Network":
    ml_model.fit(X_train, y_train)
  else:
    ml_model.fit(X_train,  y_train,epochs=100, verbose=False)


  # Make predictions
  y_pred_train = ml_model.predict(X_train)
  y_pred_test = ml_model.predict(X_test)


  # Threshold for Neural Network
  if ml_model_name == "Neural Network":
    y_pred_train = (y_pred_train > 0.5).astype(int)
    y_pred_test = (y_pred_test > 0.5).astype(int)


  # Display confusion matrices
  cm_train = display_confusion_matrix(y_train, y_pred_train, ml_model_name, "Train")
  cm_test = display_confusion_matrix(y_test, y_pred_test, ml_model_name, "Test")

  # Calculate metrics for both train and test data
  auc_roc_train, auc_roc_test = calculate_metrics(y_train, y_test, y_pred_train, y_pred_test)

  # Record the results in a data frame
  df = load_info(df,ml_model_name,'Train', balance, X_train,bankrupt_companies,cm_train, auc_roc_train)
  df = load_info(df,ml_model_name,'Test', balance, X_train,bankrupt_companies,cm_test, auc_roc_test)


  return df

# Function to balance data
def balanceData(y_train,X_train,X_test,y_test):

  # Calculate the number of samples needed from the minority class
  minority_samples = len(y_train[y_train == 1])
  majority_samples = minority_samples * 3 #desired ratio

  # Get the indices of the majority class (Healthy)
  majority_indices = np.where(y_train == 0)[0]

  # Randomly select a subset of majority samples to achieve the desired ratio
  selected_majority_indices = np.random.choice(majority_indices, majority_samples, replace=False)

  # Calculate the indices of the majority class samples that were not selected
  unselected_majority_indices = np.setdiff1d(majority_indices, selected_majority_indices)

  # Combine the minority and selected majority indices
  selected_indices = np.concatenate((np.where(y_train == 1)[0], selected_majority_indices))

  # Combine the unselected majority class samples with the test set
  X_test = np.concatenate((X_test, X_train[unselected_majority_indices]))
  y_test = np.concatenate((y_test, y_train[unselected_majority_indices]))

  # Update X_train and y_train with the selected indices
  X_train = X_train[selected_indices]
  y_train = y_train[selected_indices]


  return y_train,X_train,X_test,y_test


#[1]Read data from excel
fileName = 'Dataset2Use_Assignment1.xlsx'
sheetName = 'Total'

sheetValues = read_excel_data(fileName,sheetName)
if sheetValues is None: sys.exit()


#[3]Check for missing values
check_missing_values(sheetValues)


index_names = sheetValues.columns[:8]

#ignore last 2 column (last column is the year)
inputData = sheetValues[sheetValues.columns[:-2]].values
#categorical values
outputData = sheetValues[sheetValues.columns[-2]]
outputData, levels = pd.factorize(outputData)

for classIdx in range(0, len(np.unique(outputData))):
 tmpCount = sum(outputData == classIdx)
 tmpPercentage = tmpCount/len(outputData)
 print(' .. class', str(classIdx), 'has', str(tmpCount), 'instances', '(','{:.2f}'.format(tmpPercentage), '%)')


class_label_distribution = sheetValues.groupby([sheetValues.columns[-1], outputData]).size().unstack().fillna(0)

# Print the distribution
print(class_label_distribution)

#Figure 1, number of healthy and bankrupt businesses by year
years = class_label_distribution.index.astype(str)
healthy = class_label_distribution[class_label_distribution.columns[0]]
bankrupt = class_label_distribution[class_label_distribution.columns[1]]

# Create a stacked bar plot
plt.bar(years, healthy, label='Healthy')
plt.bar(years, bankrupt, label='Bankrupt', bottom = healthy)

# Add labels and legend
plt.xlabel('Years')
plt.ylabel('Count')
plt.title('Figure 1')
plt.legend()

# Display the plot
plt.show()

#[2.b]###########################################
#compute the min,max,avrg for each index(column)
min_values = sheetValues[sheetValues.columns[:8]].groupby([outputData]).min()
max_values = sheetValues[sheetValues.columns[:8]].groupby([outputData]).max()
avrg_values = sheetValues[sheetValues.columns[:8]].groupby([outputData]).mean()

# Calculate the y-axis range based on min and max values
y_min = min(min(min_values.loc[0].values), min(max_values.loc[0].values))
y_max = max(max(min_values.loc[0].values), max(max_values.loc[0].values))

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplots
create_subplot(ax1, min_values.loc[0],max_values.loc[0],avrg_values.loc[0], index_names,'Healthy')
create_subplot(ax2, min_values.loc[1],max_values.loc[1],avrg_values.loc[1], index_names,'Bankrupt')

# Adjust the layout
plt.tight_layout()

# Display the combined figure with subplots
plt.show()



#[4]Normalize data in [0,1]
scaler = MinMaxScaler()
inputData = scaler.fit_transform(inputData)

#[5]Stratified K-Fold with 4 folds
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

#[7.a]Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
#[7.b]Logistic Regression
logreg = LogisticRegression()
#[7.c]Decision Trees
clf = DecisionTreeClassifier()
#[7.d]Random Forests
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#[7.e]k-Nearest Neighbors
knn = KNeighborsClassifier()
#[7.f]Naïve Bayes
gnb = GaussianNB()
#[7.g]Support Vector Machines
svm = SVC()

#[7.h]Neural Network
CustomModel = keras.models.Sequential()

# Input layer
CustomModel.add(keras.layers.Input(shape=(11,)))
#CustomModel.add(keras.layers.Dense(X_train.shape[1], input_dim=11,activation='relu'))
CustomModel.add(keras.layers.Dense(64, activation='relu'))
CustomModel.add(keras.layers.Dense(64, activation='relu'))
CustomModel.add(keras.layers.Dense(32, activation='relu'))

CustomModel.add(keras.layers.Dense(1, activation='sigmoid'))

# Compile model using [accuracy, recall] to measure model performance
CustomModel.compile(optimizer='adam', loss='binary_crossentropy',metrics=[tf.keras.metrics.Accuracy(),tf.keras.metrics.Recall()])

###########


# Create an empty DataFrame with the required columns
columns = [
    'Classifier Name',
        'Set Type',
        'Balance',
        'N training samples',
        'N non-healthy companies in training sample',
        'TP',
        'TN',
        'FP',
        'FN',
        'ROC-AUC'
]
results_df = pd.DataFrame(columns=columns)

fold_n = 1
balance = "Unbalanced"
#[6]
for train_index, test_index in kfold.split(inputData, outputData):
    print("\nFold ",fold_n)

    X_train, X_test = inputData[train_index], inputData[test_index]
    y_train, y_test = outputData[train_index], outputData[test_index]

    # Put the next 2 lines in comment for unbalanced data
    y_train,X_train,X_test,y_test = balanceData(y_train,X_train,X_test,y_test)
    balance = "Balanced"

    #Count labels in train set
    train_label_counts = np.bincount(y_train, minlength=2)

    #Count labels in test set
    test_label_counts = np.bincount(y_test, minlength=2)

    print("Train Set - Healthy: ", train_label_counts[0])
    print("Train Set - Bankrupt: ", train_label_counts[1])

    print("Test Set - Healthy: ", test_label_counts[0])
    print("Test Set - Bankrupt: ", test_label_counts[1])
    print("---")

    #[7.a] Linear Discriminant Analysis model
    results_df = train_and_evaluate_classifier(results_df,lda, X_train, y_train, X_test, y_test, "LDA", balance,train_label_counts[1])
    #[7.b] Logistic Regression model
    results_df = train_and_evaluate_classifier(results_df,logreg, X_train, y_train, X_test, y_test, "LogReg", balance,train_label_counts[1])
    #[7.c] Decision Trees model
    results_df = train_and_evaluate_classifier(results_df,clf, X_train, y_train, X_test, y_test, "DTrees", balance,train_label_counts[1])
    #[7.d] Random Forest model
    results_df = train_and_evaluate_classifier(results_df,rf_classifier, X_train, y_train, X_test, y_test, "RForest", balance,train_label_counts[1])
    #[7.e] Knn model
    results_df = train_and_evaluate_classifier(results_df,knn, X_train, y_train, X_test, y_test, "KNN", balance,train_label_counts[1])
    #[7.f] Naive Bayes model
    results_df = train_and_evaluate_classifier(results_df,gnb, X_train, y_train, X_test, y_test, "Naive Bayes", balance,train_label_counts[1])
    #[7.g] Support Vector Machines model
    results_df = train_and_evaluate_classifier(results_df,svm, X_train, y_train, X_test, y_test, "SVM", balance,train_label_counts[1])

    #[7.h] Neural Network
    results_df = train_and_evaluate_classifier(results_df,CustomModel, X_train, y_train, X_test, y_test, "Neural Network", balance,train_label_counts[1])


    fold_n += 1

#print(results_df)

#[9] 
#Change name for unbalanced data
results_df.to_csv('balancedDataOutcomes.csv', index=False)
