import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


data = {
    "Credit_Score": [750, 680, 720, 640, 800, 600, 710, 780, 650, 730, 620, 820, 690, 760, 670, 790, 630, 740],
    "Income_Level": [50000, 40000, 55000, 30000, 80000, 20000, 45000, 60000, 35000, 50000, 25000, 70000, 40000, 55000, 30000, 75000, 30000, 45000],
    "Employment_Status": [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    "Default": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
}
df = pd.DataFrame(data)

X = df[["Credit_Score", "Income_Level", "Employment_Status"]]
y = df["Default"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)  # Use scaled X

model = LogisticRegression()
model.fit(X_train, y_train)

coefficients = model.coef_[0]
intercept = model.intercept_[0]
feature_names = X.columns  


print("Intercept:", intercept)
print("Coefficients:")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")


y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

sample_applicant = np.array([[700, 45000, 1]])
sample_applicant_scaled = scaler.transform(sample_applicant)

probability_default = model.predict_proba(sample_applicant_scaled)[0, 1]
print(f"Probability of Default for Sample Applicant: {probability_default:.2f}")
