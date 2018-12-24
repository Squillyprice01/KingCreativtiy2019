import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

import numpy

#parameters
willsFilePath = "C:\\Users\\Will\\Documents\\GitHub\\KingCreativtiy2019\\output.csv"
elyssasFilePath = ""
data_file = "student-mat2.csv"

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
#data_frame.to_csv(willsFilePath)
#######################################################################################################

################################## train the data set #################################################
# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs'))])

#training data. Note that this attribute cannot be one of the features in the preprocessor above.
X = training_data.drop('school', axis=1)
#training targets. Note that this attribute cannot be one of the features in the preprocessor above.
y = training_data['school']

#split data into training and testing portions for both data and targets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#fit the data
pipeline.fit(X_train, y_train)
print("model score: %.3f" % pipeline.score(X_test, y_test))
#######################################################################################################
