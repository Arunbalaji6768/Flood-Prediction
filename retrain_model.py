import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("flood_training_data.csv")

# Features and target
X = df.drop("Flood", axis=1)
y = df["Flood"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train three models
print("🔄 Training Ensemble Models...")

# 1. XGBoost
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)
print("✅ XGBoost trained")

# 2. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)
print("✅ Random Forest trained")

# 3. Logistic Regression with Polynomial Features
logistic_poly = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('logreg', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
])
logistic_poly.fit(X_train_scaled, y_train)
print("✅ Logistic Regression with Polynomial Features trained")

# Save all models and scaler
model_package = {
    'xgb_model': xgb_model,
    'rf_model': rf_model,
    'logistic_poly': logistic_poly,
    'scaler': scaler
}

joblib.dump(model_package, "model/flood_model.pkl")
print("✅ Ensemble models saved to model/flood_model.pkl")

# Evaluate performance
from sklearn.metrics import accuracy_score

models = {
    'XGBoost': xgb_model,
    'Random Forest': rf_model, 
    'Logistic Poly': logistic_poly
}

print("\n📊 Model Performance:")
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: {accuracy:.4f}")

# Ensemble prediction (weighted average)
xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
logistic_prob = logistic_poly.predict_proba(X_test_scaled)[:, 1]

ensemble_prob = (xgb_prob * 0.4 + rf_prob * 0.4 + logistic_prob * 0.2)
ensemble_pred = (ensemble_prob > 0.5).astype(int)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

print(f"🎯 Ensemble Accuracy: {ensemble_accuracy:.4f}")