import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import numpy

#parameters
willsFilePath = "C:\\Users\\Will\\Documents\\GitHub\\KingCreativtiy2019\\output.csv"
elyssasFilePath = ""
data_file = "student-mat2.csv"

#create preprocessing pipelines for both numeric and categorical data
#https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
numeric_features = ['absences']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['school','sex','address','famsize',
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
                        

#read in the csv file
training_data = pd.read_csv(data_file)

#write the processed data to a CSV 
processed_data = preprocessor.fit_transform(training_data).toarray()[:395]
data_frame = pd.DataFrame(processed_data)
data_frame.to_csv(willsFilePath)
