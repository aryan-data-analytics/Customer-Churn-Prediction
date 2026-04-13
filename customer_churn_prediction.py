import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Sample dataset
data = {
    "Age": [22, 45, 32, 36, 52, 23, 40, 60, 48, 33],
    "MonthlyCharges": [500, 1200, 800, 950, 1500, 400, 1100, 1700, 1300, 700],
    "Tenure": [3, 24, 12, 18, 36, 2, 20, 48, 30, 10],
    "ContractType": ["Monthly", "Yearly", "Monthly", "Yearly", "Yearly", "Monthly", "Monthly", "Yearly", "Yearly", "Monthly"],
    "Churn": ["Yes", "No", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes"]
}

df = pd.DataFrame(data)

# Encode categorical columns
le_contract = LabelEncoder()
le_target = LabelEncoder()

df["ContractType"] = le_contract.fit_transform(df["ContractType"])
df["Churn"] = le_target.fit_transform(df["Churn"])  # Yes=1, No=0

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", feature_importance)
