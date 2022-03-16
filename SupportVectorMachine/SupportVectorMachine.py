import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#classification data set
cancer = datasets.load_breast_cancer()
print(cancer.feature_names)
print(cancer.target_names)

X = cancer.data
Y = cancer.target

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

print(X_train, Y_train)
classes = ['malignant', 'benign']

#classifier; SVC = Support Vector Classification
#clf = svm.SVC() #-> you can use it without parameters
#clf  = svm.SVC(kernel="poly", degree=2)
#C = is the soft margin, by default it is 1, if you set it to 0 then you get a hard margin
clf  = svm.SVC(kernel="linear", C=2)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, Y_pred)
print(accuracy)
