import csv
from collections import defaultdict

columns = defaultdict(list) # each value in each column is appended to a list

with open('student-mat2.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k

print(columns['school'])

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# define example
data = columns['school']
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)


#import csv
#reader =csv
#reader = csv.reader(open('student-mat2.csv', 'r'))
#d = {}
#for row in reader:
#   k, v = row
#   d[k] = v
#print(row)





#import pandas as pd 
#colnames = ['school', 'sex', 'age']
#data = pandas.read_csv('student-mat2.csv', names=colnames)
#schools = data.school.tolist()
#print(*schools, sep = ", ") 

#df = pandas.read_csv(student-mat2.csv)
#saved_column = df.column_name #you can also use df['column_name']
#Col1 = df.school
#print(Col1)
