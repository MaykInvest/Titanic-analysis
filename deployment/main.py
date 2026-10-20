"""
FastAPI service for the exported Titanic survival model.

Your joblib file (titanic_model.pkl) already contains the FULL pipeline:
preprocessor (imputer + scaler + encoder) + tuned Gradient Boosting model.
You do NOT need to export anything else separately.

What this file adds: the feature engineering (Title, FamilySize, IsAlone,
AgeGroup) that happened in pandas BEFORE your pipeline in the notebook.
The pipeline itself never learned how to do this -- it only knows how to
impute/scale/encode the columns it was trained on. So raw passenger input
has to go through this same feature engineering first.

Run with:
    uvicorn main:app --reload
"""

import re
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Titanic Survival Prediction API")

# Load the full pipeline once, at startup (not on every request)
model = joblib.load("titanic_model.pkl")

TITLE_MAPPING = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss", "Lady": "Rare",
    "Countess": "Rare", "Capt": "Rare", "Sir": "Rare",
    "Jonkheer": "Rare", "Dona": "Rare", "Don": "Rare",
}

AGE_BINS = [0, 12, 18, 35, 60, float("inf")]
AGE_LABELS = ["Child", "Teenager", "Adult", "MiddleAged", "Senior"]


class Passenger(BaseModel):
    """Raw-ish passenger input -- close to what you'd actually have on hand."""
    name: str = Field(..., example="Braund, Mr. Owen Harris")
    pclass: int = Field(..., ge=1, le=3, example=3)
    sex: str = Field(..., example="male")
    age: float | None = Field(None, example=22.0)
    sibsp: int = Field(0, example=1)
    parch: int = Field(0, example=0)
    fare: float = Field(..., example=7.25)
    embarked: str | None = Field(None, example="S")


def engineer_features(p: Passenger) -> pd.DataFrame:
    """Replicates the exact feature engineering from the notebook."""
    # Title -- extracted the same way as df2['Name'].str.extract(...)
    match = re.search(r",\s*([^\.]+)\.", p.name)
    raw_title = match.group(1).strip() if match else "Mr"
    title = TITLE_MAPPING.get(raw_title, "Rare")

    family_size = p.sibsp + p.parch + 1
    is_alone = int(family_size == 1)

    age_group = None
    if p.age is not None:
        age_group = pd.cut([p.age], bins=AGE_BINS, labels=AGE_LABELS, right=True)[0]

    row = {
        "pclass": p.pclass,
        "sex": p.sex,
        "age": p.age,
        "sibsp": p.sibsp,
        "parch": p.parch,
        "fare": p.fare,
        "embarked": p.embarked,
        "title": title,
        "familysize": family_size,
        "isalone": is_alone,
        "agegroup": age_group,
    }
    # Must match X_train.columns exactly -- same order the pipeline was trained on
    return pd.DataFrame([row], columns=[
        "pclass", "sex", "age", "sibsp", "parch", "fare",
        "embarked", "title", "familysize", "isalone", "agegroup",
    ])


@app.post("/predict")
def predict(passenger: Passenger):
    X = engineer_features(passenger)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]

    return {
        "survived": bool(prediction),
        "survival_probability": round(float(probability), 4),
    }


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Titanic model API is running."}
