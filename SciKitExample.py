#First try at this
import csv
with open('student-mat2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1
    print(f'Processed {line_count} lines.')
#training_set = {'Dog':[[1,2,2],[2,3,2],[3,1,2]], 'Cat':[[11,20,2],[14,15,2],[12,15,2]]}
#testing_set = [15,20]
#testing_set = [1,2,2]
#
#ploting all data
#import matplotlib.pyplot as plt
#c = 'x'
#for data in training_set:
#    print(data)
    
    #print(training_set[data])
 #   for i in training_set[data]:
  #      plt.plot(i[0], i[1], c, color='c')
    
   # c = 'o'
#plt.show()

#prepare X and Y
#x = []
#y = []
#for group in training_set:
    
 #   for features in training_set[group]:
  #      x.append(features)
   #     y.append(group)

#print(x)
#print(y)

#import model builing
#from sklearn import preprocessing, neighbors
#from sklearn import linear_model

#initialize and fit

#clf = neighbors.KNeighborsClassifier() # Try different models
#clf.fit(x, y)
#reg = linear_model.LinearRegression()
#reg.fit (x,y)


#preprocess testing data
#import numpy as np
#testing_set = np.array(testing_set)
#testing_set = testing_set.reshape(1,-1)

#predition 
#prediction = reg.predict(testing_set)
#print(prediction)
