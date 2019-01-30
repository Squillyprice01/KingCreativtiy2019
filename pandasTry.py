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
                        'health','absences']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

#create a column transformer that can one hot encode categorical data and
#median scale numerical data
preprocessor= ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


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
X = training_data.drop('school', axis=1)
#training targets. Note that this attribute cannot be one of the features in the preprocessor above.
y = training_data['school']

#split data into training and testing portions for both data and targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


#fit the data
pipeline.fit(X_train, y_train)
y_pred = pipeline.fit(X_train, y_train).predict(X_test)

print("model score: %.3f" % pipeline.score(X_test, y_test))

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes='sex',
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes='sex', normalize=True,
                      title='Normalized confusion matrix')


plt.show()
#######################################################################################################
