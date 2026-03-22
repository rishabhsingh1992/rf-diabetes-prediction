# Diabetes Prediction API

## Project Purpose
This project provides a machine-learning-powered API that predicts diabetes risk from clinical patient measurements. It is designed to turn a trained Random Forest classifier into a simple prediction endpoint that can be integrated into applications, prototypes, or internal healthcare workflows.

## Core Functionality
- Trains a Random Forest classification model on diabetes dataset features.
- Standardizes input features using `StandardScaler` before model training and inference.
- Persists the trained model and scaler as reusable artifacts in the `models/` directory.
- Exposes a FastAPI `POST /predict` endpoint for real-time predictions.
- Accepts structured patient input via Pydantic validation and returns a binary outcome label.

## System Overview
The project is organized around two runtime phases:

1. Training phase (`main.py`)
- Loads the diabetes dataset.
- Splits data into train and test sets.
- Fits a feature scaler on training data.
- Trains a `RandomForestClassifier` on scaled features.
- Evaluates with test accuracy and cross-validation.
- Saves model and scaler artifacts to `models/rf_diabetes_model.pkl` and `models/rf_diabetes_scaler.pkl`.

2. Serving phase (`api/routes/predict.py`)
- Loads the serialized model and scaler once at startup.
- Validates incoming request payloads with a `PatientData` schema.
- Applies the saved scaler to incoming feature vectors.
- Runs model inference and maps the output to `Diabetic` or `Non-Diabetic`.

## Installation & Usage
1. Clone the repository and move into the project directory.
2. Create and activate a virtual environment.
3. Install dependencies from `requirements.txt`.
4. Train or refresh model artifacts by running `main.py`.
5. Start the FastAPI app.
6. Send a `POST` request to `/predict` with patient feature values.

### Example commands
```bash
# 1) Create and activate virtual environment
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train and export model artifacts
python main.py

# 4) Run API (if predict.py is the app entry)
uvicorn api.routes.predict:app --reload
```

### Example request body
```json
{
  "pregnancies": 2,
  "glucose": 120,
  "blood_pressure": 70,
  "skin_thickness": 20,
  "insulin": 79,
  "bmi": 27.5,
  "diabetes_pedigree_function": 0.45,
  "age": 33
}
```

## Notes
- Ensure the model files exist in `models/` before calling the prediction endpoint.
- The dataset path in `main.py` should be adjusted if your local directory layout differs.
