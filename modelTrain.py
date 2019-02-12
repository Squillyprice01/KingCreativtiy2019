import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np


#Use a logisitic regression
#('classifier', LogisticRegression(solver='lbfgs'))])
#Use a support vector machine
#https://scikit-learn.org/stable/modules/svm.html#classification
#('classifier', svm.SVC(gamma='scale'))])
#Use Stochastic Gradient Descent
#https://scikit-learn.org/stable/modules/sgd.html#classification
#('classifier', SGDClassifier(max_iter=1000000, tol= 0.000001))])

filePath = os.path.dirname(os.path.realpath(__file__)) + "\\student-mat2.csv"
trainingData = pd.read_csv(filePath)

#REMOVED INTERNET, TODO: automatically remove this one
categoricalFeatures = ['health','address','famsize',
                        'Pstatus','Fedu','Mjob','Fjob',
                        'reason','guardian','traveltime',
                        'studytime','failures','schoolsup',
                        'famsup','paid','activities','nursery',
                        'higher','romantic','famrel',
                        'freetime','goout','Dalc','Walc',
                        'school']

numericFeatures = ['absences']

def createPipeline(numericFeatures, categoricalFetures, classifier):
    #create preprocessing pipelines for both numeric and categorical data
    #https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html


    numeric_features = numericFeatures
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = categoricalFetures
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])


    preprocessor= ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', classifier)])

    return pipeline

def modelTrain(pipeline, featureToPredict, trainingData):

    X = trainingData.drop(featureToPredict, axis=1)
    y = trainingData[featureToPredict]

    #split data into training and testing portions for both data and targets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    #fit the data
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.fit(X_train, y_train).predict(X_test)

    return y_test, y_pred

pipeline = createPipeline(numericFeatures, categoricalFeatures, LogisticRegression(solver='lbfgs'))
y_test, y_pred = modelTrain(pipeline, 'internet', trainingData)

print(y_test)
