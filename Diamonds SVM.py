import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
import seaborn as sns

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm_df = pd.DataFrame(cm,
                     index=['0', '1', '2', '3', '4', '5', '6', '7'],
                     columns=['0', '1', '2', '3', '4', '5', '6', '7'])
plt.figure(figsize=(7,5))
sns.heatmap(cm_df, annot=True ,fmt=".1f")
plt.title('Linear SVM')
plt.ylabel('Actual Class (y_test)')
plt.xlabel('Predicted Class (y_pred)')
plt.show()
from sklearn.metrics import accuracy_score
print("Accuracy of SVM: ", accuracy_score(y_test, y_pred))
