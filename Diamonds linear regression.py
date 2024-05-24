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

# split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# perform feature scaling on the training and testing data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# fit the linear regression model to the training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict the target variable for the testing data
y_pred = regressor.predict(X_test)

# evaluate the performance of the linear regression model using mean squared error
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error: ", mse)
print("R2 Score: ", r2)
# plot the actual vs predicted target variable
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

