__author__ = 'lucas'

import numpy as np
import pandas as pd
from LucasBirch import Birch


# mean1 = [10, 10]
# mean2 = [20, 20]
# mean3 = [30, 30]
# mean4 = [40, 40]
# mean5 = [50, 50]
# cov1 = [[2.5, 0], [0, 2.5]]
# cov2 = [[1, 0], [0, 1]]
# X1= np.random.multivariate_normal(mean1, cov1, 9)
# X2= np.random.multivariate_normal(mean2, cov1, 1)
# #X3= np.random.multivariate_normal(mean2, cov1, 1)
# # X4 = np.random.multivariate_normal(mean4, cov2, n)
# # X5 = np.random.multivariate_normal(mean5, cov2, n)
# X = np.vstack((X1, X2))
# np.save('test_array', X)

n = 8

X = np.load('test_array.npy')
X = X[:n]
print(X.shape)
df = pd.DataFrame(X)
df = df.iloc[:n]

threshold = 0.1
brc = Birch(threshold, 'd1', 'r', 2)
brc.add_pandas_data_frame(df)
#brc.add_pandas_data_frame(df3)
# brc.add_pandas_data_frame(df4)
# brc.add_pandas_data_frame(df5)
print(brc.labels)

import matplotlib.pyplot as plt
labels = brc.labels[np.argsort(brc.labels[:,0]),1].astype(np.int32)
unique_labels = brc.unique_labels
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
centers = brc.centers
print centers


plt.plot(centers[:, 0], centers[:, 1], 'x')
for center, label in zip(centers, range(max(labels) + 1)) :
    print center
    class_member_mask = (labels == label)
    X_class = X[class_member_mask]
    radius = 0
    for member in X_class:
        distance = np.linalg.norm(member - center)
        if distance > radius:
            radius = distance
    print radius
    circle = plt.Circle(center,radius,color='r',fill=False)
    plt.gca().add_artist(circle)
for label, col in zip(unique_labels, colors):
    class_member_mask = (labels == label)
    X_class = X[class_member_mask]
    plt.plot(X_class[:, 0], X_class[:, 1], 'o', markerfacecolor=col)
plt.show()