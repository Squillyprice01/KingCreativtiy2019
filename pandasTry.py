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

pipeline = createPipeline(numericFeatures, categoricalFeatures, classifier)

def model(pipeline, trainingData, featureToMeasureBias, featureToPredict):

    X = trainingData.drop(featureToPredict, axis=1)
    featureToPredictList = trainingData[featureToPredict]
    featureToMeasureBiasList = trainingData[featureToMeasureBias]

    X_train, X_test, y_train, y_test = train_test_split(X, featureToPredictList, test_size=0.25)

    predictorClasses = featureToMeasureBiasList.unique()
    PredictorX_tests = []
    for i in range(len(predictorClasses)):
        PredictorX_tests.append([])
    for i in range(len(predictorClasses)):
        PredictorX_tests[i] = X_test[X_test[featureToMeasureBias] == predictorClasses[i]]

    #fit the data
    pipeline.fit(X_train, y_train)


    predictor_prediction_pairs = []
    for index in y_test.index:
        predictor_prediction_pairs.append((featureToMeasureBiasList[index], featureToPredictList[index]))

    #number of unique classes of the category whose bias we are examining  
    classPredicitons = []

    for i in range(len(predictorClasses)):
        classPredicitons.append([])

    #predictor is the category who may be biased. In our example, the sex of the person.
    #prediction is what the model predicted for the given person. In our example, the health of the person.
    for predictor, prediction in predictor_prediction_pairs:
        for i in range(len(predictorClasses)):
            if predictor == predictorClasses[i]:
                classPredicitons[i].append(prediction)
                
    for i in range(len(predictorClasses)):
        print(predictorClasses[i]," model score: %.3f" % pipeline.score(PredictorX_tests[i], classPredicitons[i]))

    return PredictorX_tests, classPredicitons

PredictorX_tests, classPredicitons = model(pipeline, trainingData, 'sex', 'health')




# dimension of the confusion matrix is the number of unique classifiers in the training data
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def confMat(testData, predData):
    #print('test',testData)
    #print('predication data',predData)
    uniqueElems = set(testData)
    #print('uniqueElems',uniqueElems)
    numUnique = len(uniqueElems)
    #print('uniqueElems', uniqueElems)
    zipping = zip(y_test,y_pred)
    pairs = list(zipping)
    counts = []
    #print('TESTING',pairs)
    for i in uniqueElems:
        for j in uniqueElems:
            mapped = list(map(lambda x : x[0] == i and x[1] == j, pairs))
            #print('MAPPED',mapped)
            count  = len(list(filter(lambda x: x, mapped)))
            #print("i =",i,", j=",j,", count=",count)
            counts.append(count)
            #print('COUNTS',counts)

    data = np.array(counts)
    #print('DATA',data)
    shape = (numUnique,numUnique)
    #print("SHAPE",shape)
    matrix = np.reshape(data,shape)
    #print('ELYSSA MATRIX')
    #print(matrix)

#confMat(y_test,y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
uniqueElems=set(y_test)
unique=list(uniqueElems)
#print(uniqueElems)
# Compute confusion matrix
#print('SCIKIT MATRIX')
#cnf_matrix = confusion_matrix(y_test, y_pred, labels=unique)
#np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=uniqueElems,
                      #title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=uniqueElems, normalize=True,
                      #title='Normalized confusion matrix')

#print(cnf_matrix)
#https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
#PrecisionFemale=
#PrecisionMale=
#plt.show()
#######################################################################################################
