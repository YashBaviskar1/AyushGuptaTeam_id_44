import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('merged_csv_final2.csv')

data.drop(['date'], axis=1, inplace=True)

X = data.drop('stability', axis=1)
# y = data['stability']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['stability'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("K-Nearest Neighbors model accuracy (in %):", accuracy_score(y_test, y_pred) * 100)
mse = mean_absolute_error(y_test, y_pred)
mae = mean_squared_error(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Random Forest Classifier model accuracy (in %):", accuracy_score(y_test, y_pred) * 100)
print("Mean Square Error : ", mse)
print("Mean Absolute Error : ", mae)