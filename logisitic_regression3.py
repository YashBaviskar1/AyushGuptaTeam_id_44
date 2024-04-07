import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import numpy as np

# Load training data
data = pd.read_csv('merged_csv_final2.csv')
data.drop(['date'], axis=1, inplace=True)
X_train = data.drop('stability', axis=1)
y_train = data['stability']

# Load test data
test_data = pd.read_csv('final-test.csv')
print(test_data.head())

X_test = test_data

reg = LogisticRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

y_true = test_data['stability']  

mse = mean_squared_error(y_true, y_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_true, y_pred)

accuracy = accuracy_score(y_true, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Accuracy:", accuracy)
