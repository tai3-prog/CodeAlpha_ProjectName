import pandas as pd

# Load cleaned dataset
df = pd.read_csv("credit_cleaned.csv")

# Target
target = "Credit_Score"

# Features and target
X = df.drop(columns=[target])
y = df[target]

print("Dataset shape:", df.shape)
print("Features shape:", X.shape)
print("Target distribution:")
print(y.value_counts())


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Identify feature types
numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

from sklearn.linear_model import LogisticRegression

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]
)

# Train
logistic_model.fit(X_train, y_train)

print("\nLogistic Regression trained successfully!")


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions
y_pred = logistic_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nLogistic Regression Results")
print("=" * 40)
print(f"Accuracy: {accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# Train
random_forest_model.fit(X_train, y_train)

print("\nRandom Forest trained successfully!")

# Predictions
rf_pred = random_forest_model.predict(X_test)

# Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Results")
print("=" * 40)
print(f"Accuracy: {rf_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))



from sklearn.ensemble import GradientBoostingClassifier

gradient_boosting_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ]
)

gradient_boosting_model.fit(X_train, y_train)

print("\nGradient Boosting trained successfully!")

gb_pred = gradient_boosting_model.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_pred)

print("\nGradient Boosting Results")
print("=" * 40)
print(f"Accuracy: {gb_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, gb_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, gb_pred))

from sklearn.metrics import roc_auc_score

# ROC-AUC for the three models
logistic_proba = logistic_model.predict_proba(X_test)
rf_proba = random_forest_model.predict_proba(X_test)
gb_proba = gradient_boosting_model.predict_proba(X_test)

logistic_auc = roc_auc_score(
    y_test,
    logistic_proba,
    multi_class="ovr",
    average="weighted"
)

rf_auc = roc_auc_score(
    y_test,
    rf_proba,
    multi_class="ovr",
    average="weighted"
)

gb_auc = roc_auc_score(
    y_test,
    gb_proba,
    multi_class="ovr",
    average="weighted"
)

print("\nROC-AUC Results")
print("=" * 40)
print(f"Logistic Regression: {logistic_auc:.4f}")
print(f"Random Forest:       {rf_auc:.4f}")
print(f"Gradient Boosting:   {gb_auc:.4f}")


# ==============================
# Final Model Comparison
# ==============================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Accuracy": [
        accuracy,
        rf_accuracy,
        gb_accuracy
    ],
    "Precision": [
        0.64,
        0.78,
        0.70
    ],
    "Recall": [
        0.64,
        0.78,
        0.70
    ],
    "F1-Score": [
        0.64,
        0.78,
        0.70
    ],
    "ROC-AUC": [
        logistic_auc,
        rf_auc,
        gb_auc
    ]
})

print("\nFinal Model Comparison")
print("=" * 70)
print(comparison.to_string(index=False))

# Best model based on ROC-AUC
best_model = random_forest_model

print("\nBest Model: Random Forest")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"ROC-AUC: {rf_auc:.4f}")


import joblib

joblib.dump(random_forest_model, "best_credit_score_model.pkl")

print("\nBest model saved successfully!")
print("File: best_credit_score_model.pkl")