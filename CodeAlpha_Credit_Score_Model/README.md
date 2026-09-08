# Credit Score Classification

## 📌 Project Overview

This project uses Machine Learning to predict an individual's **Credit Score** based on their financial and credit history.

The goal is to build and compare multiple classification models, evaluate their performance using standard classification metrics, and select the best-performing model for future predictions.

---

## 🎯 Objective

The main objectives of this project are:

* Clean and preprocess financial data.
* Perform feature engineering and prepare features for Machine Learning.
* Train multiple classification algorithms.
* Evaluate the models using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * ROC-AUC
* Compare the models and select the best-performing one.
* Use the trained model to predict the credit score of new customers.

---

## 📊 Dataset

The dataset contains financial and credit-related information for customers.

The target variable is:

`Credit_Score`

Possible classes:

* `Poor`
* `Standard`
* `Good`

The cleaned dataset contains **100,000 records** and **24 input features**.

### Main Features

The dataset includes information such as:

* Age
* Annual Income
* Monthly Inhand Salary
* Number of Bank Accounts
* Number of Credit Cards
* Interest Rate
* Number of Loans
* Delayed Payments
* Outstanding Debt
* Credit Utilization Ratio
* Total EMI per Month
* Monthly Balance
* Credit History
* Credit Mix
* Payment Behaviour
* And other financial attributes.

---

## 🧹 Data Preprocessing

The data preprocessing pipeline includes:

* Handling missing values.
* Separating numerical and categorical features.
* Numerical feature imputation using the median.
* Categorical feature imputation using the most frequent value.
* Standardizing numerical features using `StandardScaler`.
* Encoding categorical features using `OneHotEncoder`.
* Splitting the data into training and testing sets.

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

`random_state = 42`

Stratified splitting was used to preserve the target class distribution.

---

## 🤖 Machine Learning Models

Three classification algorithms were trained and compared:

### 1. Logistic Regression

A linear classification model used as a baseline model.

### 2. Random Forest

An ensemble learning algorithm based on multiple decision trees.

### 3. Gradient Boosting

An ensemble method that builds models sequentially to improve prediction performance.

---

## 📈 Model Evaluation

The models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

| Model               |   Accuracy | Precision |  Recall | F1-Score |    ROC-AUC |
| ------------------- | ---------: | --------: | ------: | -------: | ---------: |
| Logistic Regression |     64.13% |       64% |     64% |      64% |     0.7833 |
| Random Forest       | **78.40%** |   **78%** | **78%** |  **78%** | **0.8964** |
| Gradient Boosting   |     70.17% |       70% |     70% |      70% |     0.8392 |

---

## 🏆 Best Model

The **Random Forest Classifier** achieved the best overall performance.

### Performance

* **Accuracy:** 78.40%
* **Precision:** 78%
* **Recall:** 78%
* **F1-Score:** 78%
* **ROC-AUC:** 0.8964

Therefore, Random Forest was selected as the final model for credit score prediction.

The trained model is saved as:

`best_credit_score_model.pkl`

---

## 🔮 Making Predictions

The project includes a prediction script that allows users to enter customer information and receive a predicted credit score.

Run:

```bash
python predict.py
```

The script asks for the customer's financial information and returns one of the following predictions:

```text
Poor
Standard
Good
```

---

## 📁 Project Structure

```text
credit-score-classification/
│
├── data_cleaning.py
├── model.py
├── predict.py
├── credit_cleaned.csv
├── best_credit_score_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    └── original_dataset.csv
```

### File Description

| File                          | Description                              |
| ----------------------------- | ---------------------------------------- |
| `data_cleaning.py`            | Cleans and preprocesses the dataset      |
| `model.py`                    | Trains and evaluates the ML models       |
| `predict.py`                  | Predicts credit score for a new customer |
| `credit_cleaned.csv`          | Cleaned dataset                          |
| `best_credit_score_model.pkl` | Saved Random Forest model                |
| `requirements.txt`            | Required Python libraries                |
| `README.md`                   | Project documentation                    |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd credit-score-classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1. Data Cleaning

```bash
python data_cleaning.py
```

This prepares the dataset and generates the cleaned data.

### 2. Train and Compare Models

```bash
python model.py
```

This trains the three models, evaluates their performance, compares the results, and saves the best model.

### 3. Make a Prediction

```bash
python predict.py
```

Enter the customer's financial information when prompted.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Machine Learning
* Classification
* Feature Engineering
* Data Preprocessing

---

## 📌 Conclusion

This project demonstrates an end-to-end Machine Learning classification workflow for predicting credit scores.

After comparing Logistic Regression, Random Forest, and Gradient Boosting, **Random Forest achieved the best performance with an accuracy of 78.40% and a ROC-AUC score of 0.8964**.

The trained model can then be used to make predictions for new customer financial profiles.

---

## 👨‍💻 Author

**Ahmed Yassen**

Machine Learning Project — Credit Score Classification
