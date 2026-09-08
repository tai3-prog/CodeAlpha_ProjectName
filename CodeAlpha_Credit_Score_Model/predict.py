import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_credit_score_model.pkl")

print("Model loaded successfully!")
print("\nEnter customer information")
print("=" * 40)

# Numerical features
customer_data = {
    "Age": float(input("Age: ")),
    "Annual_Income": float(input("Annual Income: ")),
    "Monthly_Inhand_Salary": float(input("Monthly Inhand Salary: ")),
    "Num_Bank_Accounts": float(input("Number of Bank Accounts: ")),
    "Num_Credit_Card": float(input("Number of Credit Cards: ")),
    "Interest_Rate": float(input("Interest Rate: ")),
    "Num_of_Loan": float(input("Number of Loans: ")),
    "Delay_from_due_date": float(input("Delay from Due Date: ")),
    "Num_of_Delayed_Payment": float(input("Number of Delayed Payments: ")),
    "Changed_Credit_Limit": float(input("Changed Credit Limit: ")),
    "Num_Credit_Inquiries": float(input("Number of Credit Inquiries: ")),
    "Outstanding_Debt": float(input("Outstanding Debt: ")),
    "Credit_Utilization_Ratio": float(input("Credit Utilization Ratio: ")),
    "Total_EMI_per_month": float(input("Total EMI per Month: ")),
    "Amount_invested_monthly": float(input("Amount Invested Monthly: ")),
    "Monthly_Balance": float(input("Monthly Balance: ")),
    "Credit_History_Months": float(input("Credit History (Months): ")),
    "Num_Loan_Types": float(input("Number of Loan Types: ")),
    "Has_Loan": float(input("Has Loan? (1 = Yes, 0 = No): ")),

    # Categorical features
    "Month": input("Month: "),
    "Occupation": input("Occupation: "),
    "Credit_Mix": input("Credit Mix: "),
    "Payment_of_Min_Amount": input("Payment of Minimum Amount: "),
    "Payment_Behaviour": input("Payment Behaviour: ")
}

# Convert input to DataFrame
customer_df = pd.DataFrame([customer_data])

# Make prediction
prediction = model.predict(customer_df)

print("\nCredit Score Prediction")
print("=" * 40)
print(f"Predicted Credit Score: {prediction[0]}")