import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("../../Datasets/diabetes.csv")

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(confusion_matrix(y_test, y_pred))

importances = model.feature_importances_

feature_importances_data = pd.DataFrame(
    {"Feature": X.columns, "Importance": importances}
)

feature_importances_data = feature_importances_data.sort_values(
    by="Importance", ascending=False
)

scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

# print(f"All scores: {scores}")
# print(f"Average Accuracy: {scores.mean() * 100:.2f}%")
# print(f"Standard Deviation: {scores.std() * 100:.2f}%")

joblib.dump(model, "models/rf_diabetes_model.pkl")
joblib.dump(scaler, "models/rf_diabetes_scaler.pkl")
