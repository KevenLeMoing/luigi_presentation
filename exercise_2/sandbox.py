import requests
import csv
import pandas as pd
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pickle


csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
# Download dataset
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=MSFT&apikey=demo&datatype=csv'
response = requests.get(csv_url)

with open('out.csv', 'w') as f:
    writer = csv.writer(f)
    for line in response.iter_lines():
        writer.writerow(line.decode('utf-8').split(','))

csv_file = open('downloaded.csv', 'wb')
csv_file.write(url_content)
csv_file.close()



# Split dataset
df = pd.read_csv('downloaded.csv', delimiter=';')
features = df.drop('quality', axis=1)
label = df['quality']
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2)
x_train.to_csv('x_train.csv')
y_train.to_csv('y_train.csv')
x_test.to_csv('x_test.csv')
y_test.to_csv('y_test.csv')

# Train
x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')
regr = linear_model.LinearRegression()
regr = regr.fit(x_train, y_train)
pickle.dump(regr, open("model.pkl","wb"))

# Evaluate
model = pickle.load(open("model.pkl","rb"))
x_test = pd.read_csv('x_test.csv')
y_test = pd.read_csv('y_test.csv')
print(model.score(x_test, y_test))
pickle.dump(regr, open("serving_model.pkl","wb"))

"""


"""
print(type(x_train))
csv_file = open('x_train.csv', 'wb')
csv_file.write(x_train)
csv_file.close()
csv_file = open('y_train.csv', 'wb')
csv_file.write(y_train)
csv_file.close()



# Train  and save model
regr = linear_model.LinearRegression()
regr.fit(features, label)
#serializing our model to a file called model.pkl
pickle.dump(regr, open("model.pkl","wb"))
# Evaluate and serve model
reg.score(X, y)