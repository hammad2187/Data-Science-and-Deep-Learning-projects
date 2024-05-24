import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
dataset = pd.read_csv(r"D:\diamonds.csv")
from sklearn.preprocessing import LabelEncoder
LabE=LabelEncoder()
dataset['clarity']=LabE.fit_transform(dataset['clarity'])
dataset['cut']=LabE.fit_transform(dataset['cut'])
dataset['color']=LabE.fit_transform(dataset['color'])

x=dataset.drop(columns=['clarity'])
y=dataset['clarity']

X = dataset.iloc[:, :-1].values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dendrogram = sch.dendrogram(sch.linkage(X_train, method='ward'))

cluster = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='ward')
y_pred = cluster.fit_predict(X_train)

plt.scatter(X_train[y_pred == 0, 0], X_train[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_train[y_pred == 1, 0], X_train[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_train[y_pred == 2, 0], X_train[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_train[y_pred == 3, 0], X_train[y_pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X_train[y_pred == 4, 0], X_train[y_pred == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(X_train[y_pred == 5, 0], X_train[y_pred == 5, 1], s = 100, c = 'yellow', label = 'Cluster 6')
plt.scatter(X_train[y_pred == 6, 0], X_train[y_pred == 6, 1], s = 100, c = 'black', label = 'Cluster 7')
plt.scatter(X_train[y_pred == 7, 0], X_train[y_pred == 7, 1], s = 100, c = 'brown', label = 'Cluster 8')

plt.title('Hierarchical Clustering')
plt.xlabel('X_train')
plt.ylabel('y_train')
plt.legend()
plt.show()

