import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
file_path = r"C:\Users\ASUS\Desktop\CSV files\student\student-mat.csv"
df = pd.read_csv(file_path, delimiter=';')
selected_features = ['studytime', 'absences', 'Medu', 'Fedu', 'G1', 'G2', 'G3', 'schoolsup', 'higher', 'internet']
df = df[selected_features]
df = pd.get_dummies(df, columns=['schoolsup', 'higher', 'internet'], drop_first=True)
X = df.drop('G3', axis=1)
y = df['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R² Score:", r2)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
feature_names = X.columns
print("\nFeature Coefficients:")
for i, coef in enumerate(model.coef_):
    print(f"Coefficient for {feature_names[i]}: {coef:.2f}")
