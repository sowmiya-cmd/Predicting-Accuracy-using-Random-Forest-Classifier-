Importing Libraries

from sklearn.tree importimport seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile
%matplotlib inline DecisionTreeClassifier
import os
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report,confusion_matrix , accuracy_score ,mean_squared_error
from math import sqrt

Loading datasets

data = pd.read_csv('../input/prediction/fer (1).csv')  
      
    # Printing the dataswet shape 
print ("Dataset Length: ", len(data)) 
print ("Dataset Shape: ", data.shape) 
      
    # Printing the dataset obseravtions 
print ("Dataset: ",data.head(10)) 
 
    # Separating the target variable 
X = data.values[:, 1:12] 
Y = data.values[:, 13] 

#Predicting Accuracy using   Random Forest  Classifier:

from sklearn.ensemble import RandomForestClassifier

def splitdataset(data): 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.5, random_state = 100)
    return X, Y, X_train, X_test, y_train, y_test 

def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = RandomForestClassifier(criterion = "gini", 
            random_state = 100,max_depth=5, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 

# Function to perform training with entropy. 
def tarin_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = RandomForestClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 7, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy  
  
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred))

def main(): 
      
    # Building Phase 
    
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
      
    # Operational Phase 
    print("Results Using Gini Index:") 
      
    # Prediction using gini 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
      
if __name__=="__main__": 
    main()
