# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load training data
train_data = pd.read_csv('/content/DSAILVA-TRAIN Data - Wheat.csv')

# Selecting features (length and width of kernel)
X_train = train_data[['Length of kernel', 'Width of kernel']]

# Elbow Method to determine the optimal number of clusters (K)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Choosing the optimal number of clusters based on the Elbow Method
optimal_k = 3  # You can adjust this based on the elbow method plot

# Applying K-means clustering with the optimal K
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_train_pred = kmeans.fit_predict(X_train)

# Visualizing the clusters
plt.scatter(X_train.iloc[y_train_pred == 0, 0], X_train.iloc[y_train_pred == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_train.iloc[y_train_pred == 1, 0], X_train.iloc[y_train_pred == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X_train.iloc[y_train_pred == 2, 0], X_train.iloc[y_train_pred == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of Wheat Varieties')
plt.xlabel('Length of Kernel')
plt.ylabel('Width of Kernel')
plt.legend()
plt.show()

# Load test data
test_data = pd.read_csv('/content/DSAILVA-TEST Data - Wheat.csv')

# Predicting clusters for test data
X_test = test_data[['Length of kernel', 'Width of kernel']]
y_test_pred = kmeans.predict(X_test)

# Adding cluster predictions to test data
test_data['Cluster'] = y_test_pred

# Writing the model outcome to a file
test_data.to_csv('model_outcome.csv', index=False)

print("Model outcome saved successfully.")