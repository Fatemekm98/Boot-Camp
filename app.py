from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI(title="Customer Churn Prediction API")

# ---------- Load Pipeline ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "churn_pipeline.pkl"), "rb") as f:
    pipeline = pickle.load(f)

# ---------- Input Schema ----------
class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# ---------- Routes ----------
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running 🚀"}

@app.post("/predict")
def predict(data: ChurnInput):
    # تبدیل JSON به DataFrame
    input_df = pd.DataFrame([data.dict()])

    prediction = int(pipeline.predict(input_df)[0])
    probability = float(pipeline.predict_proba(input_df)[0][1])

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 3)
    }
