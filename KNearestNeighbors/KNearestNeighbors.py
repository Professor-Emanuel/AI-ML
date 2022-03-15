import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder() #take the labels and encode them into appropriate integer values, so we don't do it in code ourselves
buying = le.fit_transform(list(data["buying"])) #get the buying column, turn it into a list and transform them into appropriate integer values
maint = le.fit_transform(list(data["maint"])) #get the maint column, turn it into a list and transform them into appropriate integer values
door = le.fit_transform(list(data["door"])) #get the door column, turn it into a list and transform them into appropriate integer values
persons = le.fit_transform(list(data["persons"])) #get the person column, turn it into a list and transform them into appropriate integer values
lug_boot = le.fit_transform(list(data["lug_boot"])) #get the lug_boot column, turn it into a list and transform them into appropriate integer values
safety = le.fit_transform(list(data["safety"])) #get the safety column, turn it into a list and transform them into appropriate integer values
cls = le.fit_transform(list(data["class"])) #get the cls column, turn it into a list and transform them into appropriate integer values
#print(buying)
#print(maint)
#print(lug_boot)

predict = "class" #used for when we will split our data

#create the X list and Y list
X = list(zip(buying, maint, door, persons, lug_boot, safety)) # zip() = creates a tuple object from what we give it
Y = list(cls)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1) #split the data 10% into test samples
#print(X_train, Y_test)

#create the classifier
model = KNeighborsClassifier(n_neighbors = 9)
model.fit(X_train, Y_train)
accuracy = model.score(X_test, Y_test)
print(accuracy)

predicted = model.predict(X_test)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", X_test[x], "Actual: ", names[Y_test[x]])

