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
parser.add_argument("bias_feature", help="The feature to whose categories will be used to split the data")
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


filePath = os.path.dirname(os.path.realpath(__file__)) + "/" + args.data_file
categoricalFeaturePath = os.path.dirname(os.path.realpath(__file__)) + "/" + args.categorical_header_file
numericFeaturePath = os.path.dirname(os.path.realpath(__file__)) + "/" + args.numeric_header_file

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

pipeline = createPipeline(numericFeatures, categoricalFeatures, classifier)

def model(pipeline, trainingData):

    X = trainingData.drop(args.feature_to_predict, axis=1)
    featureToPredictList = trainingData[[args.feature_to_predict]]
    featureToMeasureBiasList = trainingData[[args.bias_feature]]
    print('we are predicting: ')
    print(featureToPredictList)

    X_train, X_test, y_train, y_test = train_test_split(X, featureToPredictList, test_size=0.25)
    #The entire data set is 395 rows by 33 columns (trainingData)
    #X_train is 75% of the dataset minus the column we want to predict, size: 296 rows by 32 columns
    #y_train is the column we are predicting, size: 296 rows by 1 column
    #X_test is 25% of the dataset we are using to test the trained model, size: 99 rows by 32 columns
    #y_test is column of true values, size: 99 rows by 1 column


    predictorClasses = featureToMeasureBiasList[args.bias_feature].unique()
    print('predictorClasses')
    #print(predictorClasses)

    #predictorClasses is [M,F]

    PredictorX_tests = [] # entire training set for males and females has two elements
    for i in range(len(predictorClasses)):
        PredictorX_tests.append([])
    for i in range(len(predictorClasses)):
        PredictorX_tests[i] = X_test[X_test[args.bias_feature] == predictorClasses[i]]

    #print('females')
    #print(PredictorX_tests[0][args.bias_feature])
    for i in range(len(predictorClasses)):
        print('number of ', predictorClasses[i], ' to test on: ',len(PredictorX_tests[i][args.bias_feature]))
    #print('males')
    #print(PredictorX_tests[1][args.bias_feature])
    #print('number of males to test on: ',len(PredictorX_tests[1][args.bias_feature]))

    pipeline.fit(X_train, y_train)
    #make prediction
    y_pred = pipeline.predict(X_test)
    #the length of y_pred is the same as that of y_test . . . YAY



    tupleList = []
    for i in range(len(y_test.index)):
        actualValue = y_test.values[i][0]
        index = y_test.index[i]
        predictedValue = y_pred[i]
        bias_feature = featureToMeasureBiasList[args.bias_feature].values[y_test.index[i]]
        tupleList.append((bias_feature,predictedValue, actualValue, index))


    predictorTuples =[]
    for i in range(len(predictorClasses)):
        predictorTuples.append([])

    for i in range(len(predictorClasses)):
        for feat in predictorClasses:
            for tuple in tupleList:
                if tuple[0] == feat:
                    predictorTuples[i].append(tuple)
    #print(predictorTuples[0])


    testingFor = 'Caucasian'
    print(testingFor)
    RaceTuples = []
    for tuple in tupleList:
        if tuple[0]== testingFor:
            RaceTuples.append(tuple)

    return RaceTuples

RaceTuples = model(pipeline, trainingData)
#print(tupleList)


# dimension of the confusion matrix is the number of unique classifiers in the training data
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def confMat(tuples):
    #print('test',testData)
    #print('predication data',predData)
    firstTup = tuples[0]
    biasFeat = firstTup[0]


    predictedValues =[]
    actualValues = []
    for tuple in tuples:
        predictedValues.append(tuple[1])
        actualValues.append(tuple[2])


    uniqueElems = set(actualValues)
    print(uniqueElems)
    numUnique = len(uniqueElems)
    zipping = zip(actualValues,predictedValues)
    pairs = list(zipping)
    #print(pairs)
    counts = []
    for i in uniqueElems:
        print(i)
        for j in uniqueElems:
            print(j)
            mapped = list(map(lambda x : x[0] == i and x[1] == j, pairs))
            #print(pairs)
            count  = len(list(filter(lambda x: x, mapped)))
            print(count)
            counts.append(count)
            print('finished inner loop')

    #print(pairs)
    print(counts)
    data = np.array(counts)
    shape = (numUnique,numUnique)
    matrix = np.reshape(data,shape)
    return matrix



ourMatrix = confMat(RaceTuples)
print('Our Matrix: ')
print(ourMatrix)

#to check if scikit generates the same confusion matrix
predictedVals =[]
actualVals = []
for tuple in RaceTuples:
    predictedVals.append(tuple[1])
    actualVals.append(tuple[2])
#print('predicted')
#print(predictedVals)
#print('actual')
#print(actualVals)
uniqueElems = set(actualVals)
print('SCIKIT MATRIX')
print(confusion_matrix(actualVals, predictedVals))
