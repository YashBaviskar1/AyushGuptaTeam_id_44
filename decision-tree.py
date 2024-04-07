import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv('merged_csv_final2.csv')

# Drop irrelevant columns
data.drop(['date'], axis=1, inplace=True)

# Split features and target variable
X = data.drop('stability', axis=1)
y = data['stability']

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Instantiate Decision Tree classifier
tree_classifier = DecisionTreeClassifier()

# Fit the classifier to the training data
tree_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = tree_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree model accuracy (in %):", accuracy * 100)
