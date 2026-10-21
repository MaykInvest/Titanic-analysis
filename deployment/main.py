"""FastAPI service for the Titanic survival model (pipeline: preprocessing + tuned Gradient Boosting)."""

import re
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Titanic Survival Prediction API")

# Load the model from this file's folder, so the app works from any directory
MODEL_PATH = Path(__file__).resolve().parent / "titanic_model.pkl"
model = joblib.load(MODEL_PATH)

TITLE_MAPPING = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
    "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss", "Lady": "Rare",
    "Countess": "Rare", "Capt": "Rare", "Sir": "Rare",
    "Jonkheer": "Rare", "Dona": "Rare", "Don": "Rare",
}

AGE_BINS = [0, 12, 18, 35, 60, float("inf")]
AGE_LABELS = ["Child", "Teenager", "Adult", "MiddleAged", "Senior"]

FEATURE_COLUMNS = [
    "pclass", "sex", "age", "sibsp", "parch", "fare",
    "embarked", "title", "familysize", "isalone", "agegroup",
]


class Passenger(BaseModel):
    """Passenger details as they appear in the original dataset."""
    name: str = Field(..., examples=["Braund, Mr. Owen Harris"])
    pclass: Literal[1, 2, 3] = Field(..., examples=[3])
    sex: Literal["male", "female"] = Field(..., examples=["male"])
    age: float | None = Field(None, ge=0, le=100, examples=[22.0])
    sibsp: int = Field(0, ge=0, examples=[1])
    parch: int = Field(0, ge=0, examples=[0])
    fare: float = Field(..., ge=0, examples=[7.25])
    embarked: Literal["S", "C", "Q"] | None = Field(None, examples=["S"])


def engineer_features(p: Passenger) -> pd.DataFrame:
    """Replicates the feature engineering from the notebook."""
    match = re.search(r",\s*([^\.]+)\.", p.name)
    raw_title = match.group(1).strip() if match else "Mr"
    title = TITLE_MAPPING.get(raw_title, "Rare")

    family_size = p.sibsp + p.parch + 1

    # Use np.nan (not None) for missing values, so the pipeline's imputers fill them
    # exactly like during training
    age = np.nan if p.age is None else p.age
    age_group = np.nan if p.age is None else str(
        pd.cut([p.age], bins=AGE_BINS, labels=AGE_LABELS, right=True)[0]
    )
    embarked = np.nan if p.embarked is None else p.embarked

    row = {
        "pclass": p.pclass,
        "sex": p.sex,
        "age": age,
        "sibsp": p.sibsp,
        "parch": p.parch,
        "fare": p.fare,
        "embarked": embarked,
        "title": title,
        "familysize": family_size,
        "isalone": int(family_size == 1),
        "agegroup": age_group,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


@app.post("/predict")
def predict(passenger: Passenger):
    X = engineer_features(passenger)
    probability = float(model.predict_proba(X)[0, 1])
    return {
        "survived": bool(model.predict(X)[0]),
        "survival_probability": round(probability, 4),
    }


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Titanic model API is running."}
