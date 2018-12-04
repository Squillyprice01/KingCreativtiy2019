
import pandas as pd
colnames = ['school', 'sex', 'age']
data = pd.read_csv('student-mat2.csv', names=colnames)
schools = data.school.tolist()
print(*schools, sep = ", ")
