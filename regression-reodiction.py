import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error


df = pd.read_excel('merged_file_avu.xlsx')
X = df[['Air temperature', 'Pressure', 'Wind speed']]
y = df['Power generated by system']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])


model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)


y_pred = model.predict(X_test_scaled).flatten()

mse = np.mean((y_test - y_pred) ** 2)
print('Mean Squared Error (Neural Network):', mse)

new_data = pd.read_excel("wind_power_gen_3months_validation_data.xlsx")
X_new = new_data[['Air temperature', 'Pressure', 'Wind speed']]

X_new_scaled = scaler.transform(X_new)

predicted_power = model.predict(X_new_scaled).flatten()

new_data['Predicted Power'] = predicted_power

new_data.to_csv('test_data_with_predictions_neural_network100Avu.csv', index=False)

