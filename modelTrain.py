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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_file", help="The name of the file containing data")
parser.add_argument("feature_to_predict", help="The feature of that data to train model to predict")
parser.add_argument("classifier", help="LogReg, SVM, or SGD")
parser.add_argument("categorical_header_file", help="CSV containing the categorical column names of the data")
parser.add_argument("numeric_header_file", help="CSV containing the numeric column names of the data")
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("-d", "--debug", help="used for debugging", action="store_true")
args = parser.parse_args()



classifier = None
if args.classifier == "LogReg":
    if args.verbose:
        print("Using Logisitic Regression")
    classifier = LogisticRegression(solver='lbfgs')
elif args.classifier == "SVM":
    if args.verbose:
        print("Using Support Vector Machine")
    classifier = svm.SVC(gamma='scale')
elif args.classifier == "SGD":
    if args.verbose:
        print("Using Gradient Decesnt")
    classifier = SGDClassifier(max_iter=1000000, tol= 0.000001)
else:
    print("ERROR: incorrect arguement for classifier")


filePath = os.path.dirname(os.path.realpath(__file__)) + "\\" + args.data_file
categoricalFeaturePath = os.path.dirname(os.path.realpath(__file__)) + "\\" + args.categorical_header_file
numericFeaturePath = os.path.dirname(os.path.realpath(__file__)) + "\\" + args.numeric_header_file

trainingData = pd.read_csv(filePath)
categoricalFeaturesFrame = pd.read_csv(categoricalFeaturePath)
numericFeaturesFrame = pd.read_csv(numericFeaturePath)

categoricalFeatures = list(categoricalFeaturesFrame.columns.values)
numericFeatures = list(numericFeaturesFrame.columns.values)

#Try and remove feature_to_predict from either numericFeatures or categoricalFeatures
featureInCategorical = False
featureInNumeric = False
try:
    categoricalFeatures.remove(args.feature_to_predict)
    featureInCategorical = True
except:
    pass

try:
    numericFeatures.remove(args.feature_to_predict)
    featureInNumeric = True
except:
    pass
if featureInCategorical == False and featureInNumeric == False:
    raise ValueError("feature_to_predict not a feature of the given dataset")
            
    

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

pipeline = createPipeline(numericFeatures, categoricalFeatures, classifier)
y_test, y_pred = modelTrain(pipeline, args.feature_to_predict, trainingData)

print(y_test)
