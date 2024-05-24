import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv(r"D:\diamonds.csv")
string_columns = df.select_dtypes(include=['object']).columns

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in string_columns:
    df[column] = le.fit_transform(df[column])

df = df.loc[:, ~df.columns.isin(string_columns)]
df.to_csv("diamonds_without_string.csv", index=False)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.70, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


pred=regressor.predict(X_test)
from sklearn.metrics import r2_score
accuracy = r2_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)