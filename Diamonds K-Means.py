import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"D:\diamonds.csv")
from sklearn.preprocessing import LabelEncoder
LabE=LabelEncoder()
dataset['clarity']=LabE.fit_transform(dataset['clarity'])
dataset['cut']=LabE.fit_transform(dataset['cut'])
dataset['color']=LabE.fit_transform(dataset['color'])
dataset = dataset.drop(columns=['cut'])
dataset = dataset.drop(columns=['color'])

x = dataset.drop(columns=['clarity'])
y=dataset['clarity']

X = dataset.iloc[:, :-1].values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters = 7, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'SI2')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'SI1')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'yellow', label = 'VS2')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'VS1')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'VVS2')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'purple', label = 'VVS1')
plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 100, c = 'pink', label = 'IF')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'lime', label = 'Centroids')
plt.title('Clusters of Diamonds')
plt.legend()
plt.show()
from sklearn.metrics import silhouette_score
silhouette_score = silhouette_score(X, y_kmeans)
print("The silhouette score is:", silhouette_score)

