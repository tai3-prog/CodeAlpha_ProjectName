
# ============================================================
# Heart Disease Classification
# Logistic Regression & Random Forest
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# 1. Load Dataset
# ============================================================

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

df = pd.read_csv(
    "processed.cleveland.data",
    sep=",",
    header=None,
    names=COLUMN_NAMES,
)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()


# ============================================================
# 2. Handle Missing Values
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

# Replace "?" with NaN
df.replace("?", np.nan, inplace=True)

# Convert columns to numeric
df["ca"] = pd.to_numeric(df["ca"])
df["thal"] = pd.to_numeric(df["thal"])

print("\nMissing values before imputation:")
print(df.isnull().sum())

# Fill missing values using the mode
df["ca"] = df["ca"].fillna(df["ca"].mode()[0])
df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

print("\nMissing values after imputation:")
print(df.isnull().sum())


# ============================================================
# 3. Create Binary Target
# ============================================================

# Original target:
# 0 = no disease
# 1-4 = presence of disease

df["target"] = (df["num"] > 0).astype(int)

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print("\nOriginal target distribution:")
print(df["num"].value_counts().sort_index())

print("\nBinary target distribution:")
print(df["target"].value_counts())


# ============================================================
# 4. Prepare Features and Target
# ============================================================

X = df.drop(columns=["num", "target"])
y = df["target"]

print("\n" + "=" * 60)
print("FEATURES AND TARGET")
print("=" * 60)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nFeatures:")
print(X.columns.tolist())


# ============================================================
# 5. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 6. Logistic Regression
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

# Standardization is important for Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
)

logistic_model.fit(X_train_scaled, y_train)

print("\nModel trained successfully!")

# Predictions
logistic_pred = logistic_model.predict(X_test_scaled)
logistic_prob = logistic_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
logistic_accuracy = accuracy_score(y_test, logistic_pred)
logistic_precision = precision_score(y_test, logistic_pred)
logistic_recall = recall_score(y_test, logistic_pred)
logistic_f1 = f1_score(y_test, logistic_pred)
logistic_roc_auc = roc_auc_score(y_test, logistic_prob)

print("\nAccuracy:", logistic_accuracy)
print("Precision:", logistic_precision)
print("Recall:", logistic_recall)
print("F1 Score:", logistic_f1)
print("ROC-AUC:", logistic_roc_auc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_pred))

print("\nClassification Report:")
print(classification_report(y_test, logistic_pred))


# ============================================================
# 7. Random Forest
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

# Random Forest does not require feature scaling
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

rf_model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Predictions
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_roc_auc = roc_auc_score(y_test, rf_prob)

print("\nAccuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)
print("ROC-AUC:", rf_roc_auc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))


# ============================================================
# 8. Compare Models
# ============================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
    ],
    "Accuracy": [
        logistic_accuracy,
        rf_accuracy,
    ],
    "Precision": [
        logistic_precision,
        rf_precision,
    ],
    "Recall": [
        logistic_recall,
        rf_recall,
    ],
    "F1 Score": [
        logistic_f1,
        rf_f1,
    ],
    "ROC-AUC": [
        logistic_roc_auc,
        rf_roc_auc,
    ],
})

print("\n")
print(comparison.to_string(index=False))


# ============================================================
# 9. Random Forest Feature Importance
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns,
).sort_values(ascending=False)

print("\nFeature importance:")
print(feature_importance)


# Plot feature importance
plt.figure(figsize=(10, 6))

feature_importance.sort_values().plot(
    kind="barh",
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# ============================================================
# 10. Predict a New Patient
# ============================================================

print("\n" + "=" * 60)
print("NEW PATIENT PREDICTION")
print("=" * 60)

new_patient = pd.DataFrame(
    [[
        55,     # age
        1,      # sex
        4,      # cp
        140,    # trestbps
        250,    # chol
        0,      # fbs
        1,      # restecg
        150,    # thalach
        0,      # exang
        1.0,    # oldpeak
        2,      # slope
        0,      # ca
        3,      # thal
    ]],
    columns=X.columns,
)

prediction = rf_model.predict(new_patient)[0]
probability = rf_model.predict_proba(new_patient)[0, 1]

print("\nPrediction:", prediction)
print("Probability of disease:", probability)

if prediction == 1:
    print("Result: Disease predicted")
else:
    print("Result: No disease predicted")
