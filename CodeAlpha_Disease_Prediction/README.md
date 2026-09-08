# Heart Disease Classification

A machine learning project for predicting the presence of heart disease using clinical patient data from the Cleveland Heart Disease dataset.

The project implements and compares two classification algorithms:

* Logistic Regression
* Random Forest Classifier

The goal is to build a simple and understandable machine learning pipeline covering data preprocessing, model training, evaluation, feature importance, and prediction on new patient data.

---

## 📌 Project Overview

Heart disease is one of the major health challenges worldwide. Machine learning can be used to analyze clinical data and identify patterns associated with the presence of heart disease.

In this project, patient features are used to train classification models that predict whether heart disease is present.

The original target variable contains multiple values representing different levels of disease presence. It is converted into a binary classification problem:

* `0` → No heart disease
* `1` → Heart disease present

---

## 📊 Dataset

The project uses the **Cleveland Heart Disease dataset**.

The dataset contains clinical features such as:

| Feature    | Description                           |
| ---------- | ------------------------------------- |
| `age`      | Age of the patient                    |
| `sex`      | Sex                                   |
| `cp`       | Chest pain type                       |
| `trestbps` | Resting blood pressure                |
| `chol`     | Serum cholesterol                     |
| `fbs`      | Fasting blood sugar                   |
| `restecg`  | Resting ECG results                   |
| `thalach`  | Maximum heart rate achieved           |
| `exang`    | Exercise-induced angina               |
| `oldpeak`  | ST depression                         |
| `slope`    | Slope of the peak exercise ST segment |
| `ca`       | Number of major vessels               |
| `thal`     | Thalassemia                           |
| `num`      | Original target variable              |

---

## 🔄 Machine Learning Workflow

The project follows these main steps:

```text
Dataset
   ↓
Data Exploration
   ↓
Missing Value Handling
   ↓
Target Transformation
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
New Patient Prediction
```

---

## 🧹 Data Preprocessing

The dataset contains missing values represented by `?`.

These values are:

1. Replaced with `NaN`
2. Converted to numeric values
3. Handled using imputation

The target variable is converted into a binary target:

```python
df["target"] = (df["num"] > 0).astype(int)
```

This transforms the original multi-class target into a binary classification problem.

---

## 🤖 Models

### 1. Logistic Regression

Logistic Regression is used as a baseline classification model.

Before training, the numerical features are standardized using:

```python
StandardScaler()
```

This helps Logistic Regression perform more effectively when features have different scales.

---

### 2. Random Forest

A Random Forest classifier is also trained:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

Random Forest does not require feature scaling and can capture nonlinear relationships between features.

---

## 📈 Model Evaluation

The models are evaluated using several classification metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Classification Report

These metrics provide a more complete view of model performance than accuracy alone.

---

## 🌳 Feature Importance

The Random Forest model provides feature importance scores.

These scores are used to identify which clinical features contributed most to the model's predictions.

A horizontal bar chart is generated to visualize the importance of each feature.

---

## 👤 New Patient Prediction

After training, the Random Forest model can be used to make a prediction for a new patient.

Example:

```python
new_patient = pd.DataFrame(
    [[
        55,
        1,
        4,
        140,
        250,
        0,
        1,
        150,
        0,
        1.0,
        2,
        0,
        3,
    ]],
    columns=X.columns,
)
```

The model returns:

* Predicted class
* Estimated probability of disease

Example output:

```text
Prediction: 0
Probability of disease: 0.30
```

A prediction of `0` means that the model predicts the absence of heart disease according to the learned patterns in the dataset.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 📁 Project Structure

```text
heart-disease-classification/
│
├── processed.cleveland.data
├── heart_disease.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd heart-disease-classification
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python heart_disease.py
```

---

## 📦 Requirements

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
scikit-learn
```

---

## ⚠️ Disclaimer

This project is intended for **educational and machine learning practice purposes only**.

The predictions generated by the models should not be considered medical diagnoses or used for clinical decision-making.

---

## 🎯 Learning Objectives

This project demonstrates practical experience with:

* Loading and exploring a real-world dataset
* Handling missing values
* Feature and target preparation
* Train/test splitting
* Feature scaling
* Classification algorithms
* Model evaluation
* Comparing machine learning models
* Feature importance analysis
* Making predictions on new data

---

## 🚀 Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Cross-validation
* More advanced preprocessing
* Additional classification algorithms
* Improved visualization
* Model deployment using a web application or API
* Handling preprocessing entirely through Scikit-learn Pipelines to prevent data leakage

---

## 👨‍💻 Author

**Ahmed Yassen**

Machine Learning / AI enthusiast

```

ده README مناسب جدًا كـ **Task 4 / beginner ML project**، ومش مبالغ فيه.

**قبل الـ GitHub مباشرة** هنحتاج بس نعمل حاجتين:
1. تتأكد إن اسم ملف الكود فعلًا `heart_disease.py` أو نغيره للاسم الحقيقي.
2. نضيف `requirements.txt`.

بعدها نعمل **GitHub setup + commit + push** ونكون قفلنا المشروع الأول.
```
