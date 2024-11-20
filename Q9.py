import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
np.random.seed(0)  
n_samples = 50
data = {
    "Location_Score": np.random.randint(5, 10, n_samples),
    "Size_SqFt": np.random.randint(700, 2000, n_samples),
    "Bedrooms": np.random.randint(1, 5, n_samples),
    "YearBuilt": np.random.randint(1980, 2023, n_samples),  # Added a feature
    "Price": np.random.randint(150000, 500000, n_samples),
}
df = pd.DataFrame(data)

X = df[["Location_Score", "Size_SqFt", "Bedrooms", "YearBuilt"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)  
rmse = np.sqrt(mse) 

print(f"R-squared: {r_squared:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

coefficients = model.coef_
intercept = model.intercept_
feature_names = X.columns

print("\nIntercept:", intercept)
print("Coefficients:")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")
