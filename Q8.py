import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

data = {
    "Email": [
        "Win a free prize now",
        "Meeting at 10am tomorrow",
        "Earn money quickly and easily",
        "Schedule for the next week",
        "Click here for a great offer",
        "Your invoice for last month",
        "Exclusive deal just for you",
        "Join our webinar on data science",
        "Congratulations! You've won!",
        "Important meeting reminder",
        "Double your income instantly",
        "Project update and next steps",
        "Limited time offer: 50% off",
        "Your account statement is ready",
        "Get rich quick!",
        "Team building activity next Friday",

    ],
    "Label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
}
df = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X_train, X_test, y_train, y_test = train_test_split(df["Email"], df["Label"], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('vect', vectorizer),
    ('svm', SVC(random_state=42))
])
parameters = {
    'svm__kernel': ['linear', 'rbf'],  
    'svm__C': [0.1, 1, 10] 
}
grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='f1') 
grid_search.fit(X_train, y_train)

best_svm = grid_search.best_estimator_
predictions = best_svm.predict(X_test)

print("Best SVM Classifier:")
print(best_svm)
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
