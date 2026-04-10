import joblib
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class PatientData(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float


model = joblib.load("models/rf_diabetes_model.pkl")
# scaler = joblib.load("models/rf_diabetes_scaler.pkl")


@app.post("/predict")
def predict_diabetes(data: PatientData):
    input_data = [
        [
            data.pregnancies,
            data.glucose,
            data.blood_pressure,
            data.skin_thickness,
            data.insulin,
            data.bmi,
            data.diabetes_pedigree_function,
            data.age,
        ]
    ]

    # input_scaled = scaler.transform(input_data)
    # prediction = model.predict(input_scaled)

    # Pipeline handles preprocessing and inference in one call.
    prediction = model.predict(input_data)

    return "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
