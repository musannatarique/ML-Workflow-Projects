import joblib
import numpy as np

# Load the trained SVM model
model = joblib.load(r"D:\000\Code\Py\ML\PKL\loan.pkl")

# Function to get user input for all features
def get_user_input():
    # Input for Gender, Married status, Education
    gender = input("Enter Gender (Male=1, Female=0): ")
    married = input("Enter Married status (Yes=1, No=0): ")
    education = input("Enter Education level (Graduate=1, Not Graduate=0): ")
    
    # Convert to integer (ensure inputs are valid 0 or 1)
    gender = int(gender)
    married = int(married)
    education = int(education)

    # Input for the remaining features
    dependents = input("Enter Dependents (0, 1, 2, or 3+ = 4): ")
    self_employed = input("Enter Self_Employed status (Yes=1, No=0): ")
    applicant_income = float(input("Enter Applicant Income: "))
    coapplicant_income = float(input("Enter Coapplicant Income: "))
    loan_amount = float(input("Enter Loan Amount: "))
    loan_amount_term = int(input("Enter Loan Amount Term (in months): "))
    credit_history = float(input("Enter Credit History (1 if no issues, 0 if issues): "))
    property_area = input("Enter Property Area (Urban=2, Semiurban=1, Rural=0): ")

    # Convert property_area to integer
    property_area = int(property_area)

    return np.array([[gender, married, int(dependents), education, int(self_employed), 
                      applicant_income, coapplicant_income, loan_amount, 
                      loan_amount_term, credit_history, property_area]])

# Get user input
user_input = get_user_input()

# Predict using the loaded model
prediction = model.predict(user_input)

# Show the result
if prediction == 1:
    print("Loan Approved!")
else:
    print("Loan Not Approved!")
   

