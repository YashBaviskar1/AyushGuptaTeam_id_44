import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('merged_csv_final2.csv')
data.drop(['date'], axis=1, inplace=True)

# Separate features (X) and target variable (y)
X = data.drop('stability', axis=1)
y = data['stability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Create LightGBM model object
lgbm = LGBMClassifier()

# Train the model using the training sets
lgbm.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = lgbm.predict(X_test)

# Compare actual response values (y_test) with predicted response values (y_pred)
print("LightGBM model accuracy (in %):", accuracy_score(y_test, y_pred) * 100)
