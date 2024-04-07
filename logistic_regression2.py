import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load training data
data = pd.read_csv('merged_csv_final2.csv')
data.drop(['date'], axis=1, inplace=True)
X = data.drop('stability', axis=1)
y = data['stability']

# Load test data
test_data = pd.read_csv('final-test.csv')

# Create and train logistic regression model
reg = LogisticRegression()
reg.fit(X, y)

# Predict stability for test data
y_pred = reg.predict(test_data)

# Map predictions to simple labels
simple_predictions = ['Stable' if pred == 'stable' else 'Unstable' for pred in y_pred]

# Create a DataFrame from simple predictions
df = pd.DataFrame({'prediction': simple_predictions})

# Save the predictions to a CSV file
df.to_csv('predictions2.csv', index=False)
