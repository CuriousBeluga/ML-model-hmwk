import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.metrics import classification_report

from IPython.display import display
# Load the CSV file
data = pd.read_csv('ai4i2020.csv')

data = data.drop(['UDI','Product ID','TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)

# Perform categorical data transformation
label_encoder = LabelEncoder()
data['Type'] = label_encoder.fit_transform(data['Type'].str.strip())

# Perform normalization
scaler = MinMaxScaler()
columns_to_normalize = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                        'Tool wear [min]']
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# Perform standardization
standard_scaler = StandardScaler()
columns_to_standardize = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
                          'Tool wear [min]']
data[columns_to_standardize] = standard_scaler.fit_transform(data[columns_to_standardize])
print(data)

# Separate the input attributes and target variable
X = data.drop('Machine failure', axis=1)
y = data['Machine failure']

# undersample the datapoints to test ML binary classifiers

undersampler = RandomUnderSampler(sampling_strategy=1)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# ----------------------------------------------------------------------------------------------------------------------
# 5-fold cross-validation of ML models

score_bot = make_scorer(matthews_corrcoef)

# result collection list
all_results = []

# ANN

ANN = MLPClassifier()
ANN_param_grid = {'hidden_layer_sizes': [(50,), (100,), (200,)],
                  'activation': ['relu', 'logistic']}

ANN_grid_search = GridSearchCV(ANN, ANN_param_grid, scoring=score_bot, cv=5)
ANN_grid_search.fit(X_train, y_train)
ANN_best_model = ANN_grid_search.best_estimator_
ANN_best_score = ANN_grid_search.best_score_
ANN_cv_scores = cross_val_score(ANN_best_model, X_train, y_train, scoring=score_bot, cv=5)
all_results.append(['Artificial Neural Networks', ANN_best_model.get_params(), ANN_best_score])  # append ANN results

# Support Vector Machine
SVM = SVC()
SVM_param_grid = {'C': [0.1, 1.0, 10.0],
                  'kernel': ['linear', 'rbf', 'sigmoid'], }

SVM_grid_search = GridSearchCV(SVM, SVM_param_grid, scoring=score_bot, cv=5)
SVM_grid_search.fit(X_train, y_train)
SVM_best_model = SVM_grid_search.best_estimator_
SVM_best_score = SVM_grid_search.best_score_
SVM_cv_scores = cross_val_score(SVM_best_model, X_train, y_train, scoring=score_bot, cv=5)
all_results.append(['Support Vector Machine', SVM_best_model.get_params(), SVM_best_score])  # append SVM results

# KNN

KNN = KNeighborsClassifier()
KNN_param_grid = {'n_neighbors': [3, 5, 7],
                  'p': [1, 2], }

KNN_grid_search = GridSearchCV(KNN, KNN_param_grid, scoring=score_bot, cv=5)
KNN_grid_search.fit(X_train, y_train)
KNN_best_model = KNN_grid_search.best_estimator_
KNN_best_score = KNN_grid_search.best_score_
KNN_cv_scores = cross_val_score(KNN_best_model, X_train, y_train, scoring=score_bot, cv=5)
all_results.append(['K-Nearest Neighbors', KNN_best_model.get_params(), KNN_best_score])  # append KNN results

# Decision Tree
dt = DecisionTreeClassifier()
dt_param_grid = {
    'criterion': ['gini', 'entropy'],
    'ccp_alpha': [0.0, 0.1, 0.2],
}
dt_grid_search = GridSearchCV(dt, dt_param_grid, scoring=score_bot, cv=5)
dt_grid_search.fit(X_train, y_train)
dt_best_model = dt_grid_search.best_estimator_
dt_best_score = dt_grid_search.best_score_
dt_cv_scores = cross_val_score(dt_best_model, X_train, y_train, scoring=score_bot, cv=5)
all_results.append(['Decision Tree', dt_best_model.get_params(), dt_best_score])  # append dt results

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_score = matthews_corrcoef(y_train, lr.predict(X_train))
all_results.append(['Linear Regression', 'Not Applicable for Linear Regression', lr_score])  # append lr results

# display
pd.set_option('display.max_colwidth', None)
all_results_df = pd.DataFrame(all_results, columns=['ML Trained Model',
                                                    'Best Set of Parameter Values',
                                                    'MCC Score for 5-fold Cross Validation'])
print(all_results_df)
# ----------------------------------------------------------------------------------------------------------------------

# Evaluate the performance of the best-trained models on the testing dataset
ann_test_predictions = ANN_best_model.predict(X_test)
svm_test_predictions = SVM_best_model.predict(X_test)
knn_test_predictions = KNN_best_model.predict(X_test)
dt_test_predictions = dt_best_model.predict(X_test)
lr_test_predictions = lr.predict(X_test)

# Calculate MCC-scores on the testing dataset
ann_test_score = matthews_corrcoef(y_test, ann_test_predictions)
svm_test_score = matthews_corrcoef(y_test, svm_test_predictions)
knn_test_score = matthews_corrcoef(y_test, knn_test_predictions)
dt_test_score = matthews_corrcoef(y_test, dt_test_predictions)
lr_test_score = matthews_corrcoef(y_test, lr_test_predictions)

mcc_scores = [ann_test_score, svm_test_score, knn_test_score, dt_test_score, lr_test_score]
model_names = ['Artificial Neural Networks', 'Support Vector Machine', 'K-Nearest Neighbors',
              'Decision Tree', 'Logistic Regression']

# Create a DataFrame to compare the MCC-scores of all models on the testing dataset
pd.set_option('display.max_colwidth', None)
test_scores = pd.DataFrame({
    'Model': model_names,
    'Best Parameters': [ANN_best_model.get_params(), SVM_best_model.get_params(), KNN_best_model.get_params(),
                        dt_best_model.get_params(), 'Not Applicable for Linear Regression'],
    'MCC-Score': mcc_scores
})

# Sort the DataFrame in descending order of MCC-Score
test_scores = test_scores.sort_values(by='MCC-Score', ascending=False)

# Print the DataFrame
print(test_scores)

# Declare the best algorithm
score_dict = {name: score for name,score in zip(model_names,mcc_scores)}
max_score = max(score_dict.values())
key_name = max(score_dict,key=score_dict.get)
print(f'{key_name} has maximum MCC score of {max_score}')

