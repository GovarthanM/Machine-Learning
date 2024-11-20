import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml

data = fetch_openml('mnist_784', version=1)
X = data.data / 255.0 
y = data.target.astype(int)

X = X[(y == 3) | (y == 8)]
y = y[(y == 3) | (y == 8)]
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('pca', PCA()), 
    ('knn', KNeighborsClassifier()) 
])
parameters = {
    'pca__n_components': [25, 50, 75], 
    'knn__n_neighbors': list(range(1, 11, 2)),
    'knn__weights': ['uniform', 'distance']
}
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, scoring='accuracy') 
grid_search.fit(X, y)  
best_knn = grid_search.best_estimator_
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
