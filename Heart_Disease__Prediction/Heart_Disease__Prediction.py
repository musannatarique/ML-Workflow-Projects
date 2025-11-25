import numpy as np
import joblib  # or pickle if you're using pickle to load the model

# Load your trained model (make sure you have your model file in the same directory or provide the correct path)
model = joblib.load(r"D:\000\Code\Py\ML\PKL\heart_disease_model.pkl")  # Change filename accordingly

# Taking user input
print("Please enter the following details:")
age = float(input("Age: "))
sex = float(input("Sex (1 = male; 0 = female): "))
cp = float(input("Chest Pain Type (0-3): "))
trestbps = float(input("Resting Blood Pressure in mm Hg (0-500): "))
chol = float(input("Serum Cholesterol (in mg/dl): "))
fbs = float(input("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false): "))
restecg = float(input("Resting Electrocardiographic results (0-2): "))
thalach = float(input("Maximum Heart Rate Achieved: "))
exang = float(input("Exercise Induced Angina (1 = yes; 0 = no): "))
oldpeak = float(input("ST depression induced by exercise: "))
slope = float(input("Slope of the peak exercise ST segment (0-2): "))
ca = float(input("Number of major vessels (0-3): "))
thal = float(input("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect): "))

# Make prediction
input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print("Prediction result:", prediction)

if prediction[0] == 0:
    print('The Person does not have Heart Disease')
else:
    print('The Person has Heart Disease')
