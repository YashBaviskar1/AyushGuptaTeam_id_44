import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv('merged_csv_final2.csv')
data.drop(['date'], axis=1, inplace=True)
X = data.drop('stability', axis=1)
y = data['stability']

df = pd.DataFrame(data)

X = df.drop('stability', axis=1)
y = df['stability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

reg = LogisticRegression()

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("Logistic Regression model accuracy (in %):", accuracy_score(y_test, y_pred) * 100)
