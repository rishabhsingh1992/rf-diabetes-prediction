import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

df = pd.read_csv("../../Datasets/diabetes.csv")

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaler = StandardScaler()
#
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# model = RandomForestClassifier(n_estimators=100, random_state=42)
#
# model.fit(X_train_scaled, y_train)
#
# y_pred = model.predict(X_test_scaled)

# Build a pipeline that keeps the same scaler + model behavior.
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
)

# Train the pipeline end-to-end on unscaled training data.
model.fit(X_train, y_train)

# Predict with identical preprocessing handled inside the pipeline.
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(confusion_matrix(y_test, y_pred))

importances = model.named_steps["classifier"].feature_importances_

feature_importances_data = pd.DataFrame(
    {"Feature": X.columns, "Importance": importances}
)

feature_importances_data = feature_importances_data.sort_values(
    by="Importance", ascending=False
)

# scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

# Cross-validation remains identical, now using pipeline-managed preprocessing.
scores = cross_val_score(model, X_train, y_train, cv=5)

# print(f"All scores: {scores}")
# print(f"Average Accuracy: {scores.mean() * 100:.2f}%")
# print(f"Standard Deviation: {scores.std() * 100:.2f}%")

joblib.dump(model, "models/rf_diabetes_model.pkl")
# joblib.dump(scaler, "models/rf_diabetes_scaler.pkl")

# Keep exporting a scaler artifact for compatibility with existing workflows.
joblib.dump(model.named_steps["scaler"], "models/rf_diabetes_scaler.pkl")
