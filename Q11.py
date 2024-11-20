import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

try:
    data = pd.read_csv(r"C:\Users\ASUS\Desktop\CSV files\cities.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("CSV file not found. Please check the file path.")
    raise

print("Columns in the dataset:", data.columns)
print("Sample data:\n", data.head())
print("Data types:\n", data.dtypes)

data.columns = data.columns.str.strip()

required_columns = ['date', 'temperature']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"Column '{col}' not found in the CSV file.")

try:
    
    data['day_of_year'] = pd.to_datetime(data['date'], errors='coerce').dt.dayofyear
    
    data.dropna(subset=['day_of_year'], inplace=True)
    data['sin_day'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
    data['cos_day'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
except Exception as e:
    print("Error during feature engineering:", e)
    raise 
if 'temperature' not in data.columns:
    raise KeyError("Column 'temperature' not found in the CSV file.")
    
X_poly_features = ['sin_day', 'cos_day']
y = data['temperature']

poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(data[X_poly_features])

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("Sample predictions:\n", results.head())
