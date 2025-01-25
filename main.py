import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from BallTree import BallTree
from KNeighborsClassifier import KNeighborsClassifier

X = np.array([
    [8, 11],
    [4, 2],
    [3, 9],
    [4, 1],
    [2, 7],
    [12, 3],
    [5, 6],
    [13, 7],
    [10, 8],
    [0, 4],
    [4, 5],
    [10, 19],
    [18, 4],
])

# Using KMeans to generate 2 clusters for label assignment
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Assign labels based on the clusters (0 or 1)
y = kmeans.labels_

plt.scatter(X[:, 0], X[:, 1], c=y)

X_test = np.array([
    [5, 7],
])

# Plot X_test in color red
plt.scatter(X_test[:, 0], X_test[:, 1], c='red')

knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', leaf_size=2, metric='euclidean')
knn.fit(X, y)

results = knn.kneighbors(X_test)
print(results[0])

plt.show()



