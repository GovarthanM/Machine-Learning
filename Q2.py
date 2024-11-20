

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic time series data (replace with your actual data)
np.random.seed(42)  # for reproducibility
dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='M')
trend = np.arange(len(dates))
seasonality = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
noise = np.random.normal(0, 5, len(dates))
electricity_consumption = trend + seasonality + noise
df = pd.DataFrame({'Date': dates, 'Consumption': electricity_consumption})


# Feature Engineering
df['Time'] = np.arange(len(df))  
df['Month'] = df['Date'].dt.month  
df = pd.get_dummies(df, columns=['Month'], drop_first=True) # One-hot encode month
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

# Model Training
X_train = train_data[['Time'] + [col for col in train_data.columns if 'Month' in col]] # Include time and month dummies
y_train = train_data['Consumption']
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
X_test = test_data[['Time'] + [col for col in test_data.columns if 'Month' in col]]
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(test_data['Consumption'], y_pred))
print(f"RMSE: {rmse}")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(test_data['Date'], test_data['Consumption'], label='Actual')
plt.plot(test_data['Date'], y_pred, label='Predicted', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Electricity Consumption')
plt.title('Electricity Consumption Prediction')
plt.legend()
plt.show()
