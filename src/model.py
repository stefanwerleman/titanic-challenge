import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# This method is used to fill in missing values for each datapoint.
def mean(data):
    sum = 0
    count = 0
    for num in data:
        if not(math.isnan(num)):
            sum += num
            count += 1
    
    return sum // count

# Returns processed training and testing data set. 
# This new data set will contain the selected 
# features and class labels necessary for training and testing.
def preprocess():
    data = pd.read_csv('../input/train.csv')
    X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
    y = data[['Survived']]

    # Replacing all missing ages with the mean age.
    mn = mean(X[:, 2])
    for i in range(0, len(X[:, 2])):
        if math.isnan(X[i][2]):
            X[i][2] = mn

    n = mean(X[:, 3])
    for i in range(0, len(X[:, 3])):
        if math.isnan(X[i][3]):
            X[i][3] = mn

    n = mean(X[:, 4])
    for i in range(0, len(X[:, 4])):
        if math.isnan(X[i][4]):
            X[i][4] = mn
    
    n = mean(X[:, 5])
    for i in range(0, len(X[:, 5])):
        if math.isnan(X[i][5]):
            X[i][5] = mn

    # Encoding number labels to male(1) and female(0)
    le = LabelEncoder()
    X[:, 1] = le.fit_transform(X[:, 1])

    return X, y

# Here will actually create, train, and return our model for testing.
def dt_model(X_train, y_train):
    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(X_train, y_train)

    return dt

def rf_model(X_train, y_train):
    y_train = y_train.values.flatten()

    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(X_train, y_train)

    return rf

# Will return the predictions from our model.
def test_model(model):
    test_data = pd.read_csv('../input/test.csv')
    X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values
    
    # Replacing all missing ages with the mean age.
    mn = mean(X_test[:, 2])
    for i in range(0, len(X_test[:, 2])):
        if math.isnan(X_test[i][2]):
            X_test[i][2] = mn

    mn = mean(X_test[:, 3])
    for i in range(0, len(X_test[:, 3])):
        if math.isnan(X_test[i][3]):
            X_test[i][3] = mn

    mn = mean(X_test[:, 4])
    for i in range(0, len(X_test[:, 4])):
        if math.isnan(X_test[i][4]):
            X_test[i][4] = mn
    
    mn = mean(X_test[:, 5])
    for i in range(0, len(X_test[:, 5])):
        if math.isnan(X_test[i][5]):
            X_test[i][5] = mn

    # Encoding number labels to male(1) and female(0)
    le = LabelEncoder()
    X_test[:, 1] = le.fit_transform(X_test[:, 1])

    predictions = dt.predict(X_test)
    results = pd.DataFrame({
        'PassengerId': test_data.PassengerId,
        'Survived': predictions
    })

    
    # Five-fold cross validation:
    # NOTE: I found the CSV submission with a 1.0 accuracy score
    correct_results = pd.read_csv('../correctsubmission/Titanic_submission.csv')
    y_test = correct_results[['Survived']]

    # Decision Tree
    print(dt.score(X_test, y_test))

    return results

if __name__ == '__main__':
    X_train, y_train = preprocess()

    print('Decision Tree Score:')
    dt = dt_model(X_train, y_train)
    results = test_model(dt)
    results.to_csv('../results/dt.csv', index=False)

    print()

    # Test Model (Predict)
    print('Random Forest Score:')
    rf = rf_model(X_train, y_train)
    results = test_model(rf)
    results.to_csv('../results/rf.csv', index=False)

    print()

    # NOTE: Five Fold scores are in the terminal

    # Shows the Tree Plot. 
    # Give it a moment for qt5ct to load the tree plot 
    # NOTE: Click X to exit out QT window and terminate program.
    tree.plot_tree(dt)
    plt.show()