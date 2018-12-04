import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy

willsFilePath = "C:\\Users\\Will\\Documents\\GitHub\\KingCreativtiy2019\\output.csv"
elyssasFilePath = ""

#read in the csv file
trainingData = pd.read_csv('student-mat2.csv')
print(trainingData)

#One hot encode that data
ohe = OneHotEncoder(sparse=False)
transformed = ohe.fit_transform(trainingData)
print(transformed)

#write the csv
df = pd.DataFrame(transformed)
df.to_csv(willsFilePath)
