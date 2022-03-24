import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
#scale down so all features are between -1 and 1 , .data is all our features
#scaling will make things faster
data = scale(digits.data)
y = digits.target
k = 10 #len(np.unique(y)) #dynamic way to do it, we know it is 10, so we can just type 10

#get the number of features that go along with that data
samples, features = data.shape

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

#make the classifier, n_init=10 is the default value
#KMeans has many parameters, but we will leave the default values
clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)
