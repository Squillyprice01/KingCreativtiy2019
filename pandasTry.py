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

#parameters
data_file = "student-mat2.csv"
file_path = os.path.dirname(os.path.realpath(__file__)) + "\\output.csv"

#read in the csv file
training_data = pd.read_csv(data_file)

#create preprocessing pipelines for both numeric and categorical data
#https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

#This array could be replaced with the column names of quantitative data for any data set
numeric_features = ['absences']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

#This array could be replaced with the column names of categorical data for any data set
categorical_features = ['sex','address','famsize',
                        'Pstatus','Fedu','Mjob','Fjob',
                        'reason','guardian','traveltime',
                        'studytime','failures','schoolsup',
                        'famsup','paid','activities','nursery',
                        'higher','internet','romantic','famrel',
                        'freetime','goout','Dalc','Walc',
                        'school','absences']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

#create a column transformer that can one hot encode categorical data and
#median scale numerical data
preprocessor= ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

#partition this data based on this column split on so that
#male vs female two conf matrices
##################################  write the processed data to a CSV##################################
#processed_data = preprocessor.fit_transform(training_data).toarray()[:training_data['school'].size]
#data_frame = pd.DataFrame(processed_data)
#data_frame.to_csv(file_path)
#######################################################################################################

################################## train the data set #################################################
# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      #Use a logisitic regression
                      #('classifier', LogisticRegression(solver='lbfgs'))])
                      #Use a support vector machine
                      #https://scikit-learn.org/stable/modules/svm.html#classification
                      #('classifier', svm.SVC(gamma='scale'))])
                      #Use Stochastic Gradient Descent
                      #https://scikit-learn.org/stable/modules/sgd.html#classification
                       ('classifier', SGDClassifier(max_iter=1000000, tol= 0.000001))])

#training data. Note that this attribute cannot be one of the features in the preprocessor above.
X = training_data.drop('health', axis=1)
#training targets. Note that this attribute cannot be one of the features in the preprocessor above.
y = training_data['health']

#X 75% and Y 25%
#split data into training and testing portions for both data and targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print('X_test',X_test)
MaleX_test = X_test[X_test['sex']== 'M']
FemaleX_test = X_test[X_test['sex']== 'F']
print('YTEST')
print(type(y_test))
indeces = y_test.index
print('INDECES',indeces)
y_testList = list(y_test)
print(y_testList[4])
print(y_testList)
malePred = []
femalePred = []

for i in y_test:
    indeces = y_test.index
    #myseries[myseries == 7].index[0]
    print(X_test[X_test ==indeces[i]])

    #if X_test[X_test == indeces[i]]['sex']=='M']]:
        #malePred.append(y_test[i])
    #if X_test[indeces[i]['sex']=='M']]:
        #malePred.append(y_test[i])
#MaleY_test = y_test[y_test['sex']== 'M']
#FemaleY_test = y_test[y_test['sex']== 'F']

# MFTest= X_test.loc[:,'sex']
#print('MALE DATA')
#print(MaleX_test)
#print(y_test[0])
#make a loop to split males and females of health
#for i in y_test:

#fit the data
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(MaleX_test)
#print('test',y_test)
#print('pred' ,y_pred)

print("model score: %.3f" % pipeline.score(MaleX_test, MaleY_test))
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
    print('ELYSSA MATRIX')
    print(matrix)

confMat(y_test,y_pred)

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
print('SCIKIT MATRIX')
cnf_matrix = confusion_matrix(y_test, y_pred, labels=unique)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=uniqueElems,
                      #title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=uniqueElems, normalize=True,
                      #title='Normalized confusion matrix')

print(cnf_matrix)
#https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
#PrecisionFemale=
#PrecisionMale=
#plt.show()
#######################################################################################################
