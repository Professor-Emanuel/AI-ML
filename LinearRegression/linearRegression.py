# In PyCharm Terminal do
# 1. activate tensor
# 2. pip install sklearn
# 3. pip install pandas
# 4. pip install matplotlib
# 5. pip install numpy

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";") #csv = comma separated values, but the data is separated by semicolons

#print(data.head())

#G1, G2, studytime, failures, absences -> attributes based on which we want to predict G3 -> attribute still here!
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#print(data.head())

predict = "G3" #G3 -> label here

X = np.array(data.drop([predict], 1)) #original array without G3
Y = np.array(data[predict]) # array just with G3
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1) #split the data 10% into test samples

'''this part is just for training models and finding the best one, for which the accuracy is the highest, uncomment->run->comment out
best = 0
#we are going to loop 30 times, but you can increase it to what ever you want
for _ in range(30):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1) #split the data 10% into test samples

    linear = linear_model.LinearRegression()

    linear.fit(X_train, Y_train) #fit the data to find the BEST FIT LINE
    accuracy = linear.score(X_test, Y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
        #create the .pickle file, write to it if the condition is true
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
'''

#open the file in read binary mode
pickle_in = open("studentmodel.pickle", "rb")
#load this pickle in our linear model
linear = pickle.load(pickle_in)

print("Coefficients: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(X_test)
for x in range(len(predictions)):
    print(predictions[x], X_test[x], Y_test[x])

#plot the data
p = 'absences' #one of the attributes/features -> choose whatever you want between G1, G2, studytime, failures, absences
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

