import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"D:\diamonds.csv")
from sklearn.preprocessing import LabelEncoder
LabE=LabelEncoder()
dataset['clarity']=LabE.fit_transform(dataset['clarity'])
dataset['cut']=LabE.fit_transform(dataset['cut'])
dataset['color']=LabE.fit_transform(dataset['color'])

X = dataset.drop(columns=['clarity'])
y=dataset['clarity']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70)

# Scale the data using StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the KNN model on the training data
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
KNN.fit(X_train, y_train)

# Make predictions on the test data
y_pred1 = KNN.predict(X_test)

# Evaluate the model's performance using accuracy score and confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm1 = confusion_matrix(y_test, y_pred1)
print("Confusion Matrix:\n", cm1)
print("Accuracy of KNN: ", accuracy_score(y_test, y_pred1))

# Plot the confusion matrix using seaborn
import seaborn as sns
cm1 = confusion_matrix(y_test, y_pred1)
cm_df = pd.DataFrame(cm1,
                     index=['0', '1', '2', '3', '4', '5', '6', '7'],
                     columns=['0', '1', '2', '3', '4', '5', '6', '7'])

plt.figure(figsize=(7, 4))
sns.heatmap(cm_df, cmap='Blues', annot=True, fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()