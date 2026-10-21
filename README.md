# 🚢 Titanic Survival Prediction

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-purple)

A classification project that predicts which passengers survived the Titanic. It covers exploratory analysis, feature engineering, a leakage-free scikit-learn pipeline, a baseline, a comparison of three models, Bayesian hyperparameter tuning and a full evaluation on held-out passengers.

---

## 📌 Results at a Glance

| | Value |
|---|---|
| **Final model** | Gradient Boosting, tuned with `BayesSearchCV` |
| **Test accuracy** | **83%** (179 unseen passengers) |
| **Test ROC-AUC** | **0.83** |
| **Cross-validation ROC-AUC** | 0.896 |
| **Baseline accuracy** | 61.7% (always predicting "did not survive") |

---

## 📦 Dataset

| Property | Detail |
|---|---|
| **Source** | [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) (training file) |
| **File** | [`data/titanic.csv`](data/titanic.csv) |
| **Size** | 891 passengers, 12 columns |
| **Target** | `Survived` (0 = did not survive, 1 = survived); 38.4% survived |
| **Missing values** | `Age` 177, `Cabin` 687, `Embarked` 2 |

---

## 🔍 What the Data Shows

Survival rates calculated from the dataset:

| Group | Survival rate | Passengers |
|---|---|---|
| Women | **74.2%** | 314 |
| Men | **18.9%** | 577 |
| 1st class | 63.0% | 216 |
| 2nd class | 47.3% | 184 |
| 3rd class | 24.2% | 491 |
| Title "Mrs" / "Miss" | 79.4% / 70.3% | 126 / 185 |
| Title "Master" (boys) | 57.5% | 40 |
| Title "Mr" | 15.7% | 517 |
| Travelling alone | 30.4% | 537 |
| Family of 2–4 | 57.9% | 292 |
| Family of 5 or more | 16.1% | 62 |
| Children (12 or under) | 58.0% | 69 |

**In short:** "women and children first" is clearly visible, passenger class made a large difference, and people in small families did better than both solo travellers and large families.

The notebook also explains the model itself with **permutation feature importance** (how much the test ROC-AUC drops when a feature is shuffled) and **SHAP values**, which show how each feature pushes a prediction towards or away from survival, overall and for individual passengers.
**Most important features for the model:** title (by far), passenger class and age (permutation importance), confirmed by SHAP, which also ranks fare highly.

---

## 🗺️ Pipeline

| Step | What was done |
|---|---|
| 1 | Load the data and check its structure, missing values and duplicates |
| 2 | Feature engineering (below) and drop `PassengerId`, `Name`, `Ticket`, `Cabin` |
| 3 | Visual EDA (age, class, sex, title and family size vs. survival, correlation heatmap) and an automated profiling report with `data-profiling` ([`Titanic_report.html`](Titanic_report.html)) |
| 4 | Stratified train/test split (80/20) |
| 5 | Preprocessing pipeline: median imputation + scaling for numbers, most-frequent imputation + one-hot encoding for categories. Missing ages are filled **inside** the pipeline, so nothing leaks from validation data |
| 6 | Baseline: `DummyClassifier` (most frequent class) |
| 7 | Compare Logistic Regression, Random Forest and Gradient Boosting with 5-fold `StratifiedKFold` |
| 8 | Tune Gradient Boosting with `BayesSearchCV` (50 iterations, scoring ROC-AUC) |
| 9 | Evaluate once on the test set: classification report, confusion matrix, ROC and precision-recall curves |
| 10 | Explain the model: permutation feature importance and SHAP values (global and per-passenger) |
| 11 | Save the fitted pipeline with `joblib` |
| 12 | Serve the model through a REST API with FastAPI ([`deployment/main.py`](deployment/main.py)) |

### Feature engineering

| Feature | Description |
|---|---|
| `title` | Extracted from the name (`Mr`, `Mrs`, `Miss`, `Master`, or `Rare` for Dr, Rev, Col, etc.) |
| `familysize` | `sibsp + parch + 1` |
| `isalone` | 1 if travelling alone |
| `agegroup` | Child (≤12), Teenager (13–18), Adult (19–35), MiddleAged (36–60), Senior (over 60) |

---

## 🤖 Model Comparison (5-fold cross-validation on the training set)

| Model | CV Accuracy | CV ROC-AUC |
|---|---|---|
| Baseline (`DummyClassifier`) | 0.617 | 0.500 |
| Logistic Regression | 0.810 | 0.862 |
| Random Forest | 0.803 | 0.866 |
| **Gradient Boosting** 🏆 | **0.820** | **0.893** |

### Hyperparameter tuning (`BayesSearchCV`)

| Parameter | Best value |
|---|---|
| `n_estimators` | 203 |
| `learning_rate` | 0.078 |
| `max_depth` | 4 |
| `min_samples_leaf` | 5 |
| `subsample` | 0.73 |

Best CV ROC-AUC: **0.896**.

---

## 📊 Final Evaluation (179 unseen passengers)

| Class | Precision | Recall | F1-score | Passengers |
|---|---|---|---|---|
| Did not survive | 0.84 | 0.89 | 0.87 | 110 |
| Survived | 0.81 | 0.74 | 0.77 | 69 |
| **Accuracy** | | | **0.83** | 179 |

| Check | ROC-AUC |
|---|---|
| Cross-validation (training) | 0.896 |
| Test set | 0.832 |
| Difference | 0.064 |

The model is better at recognising passengers who did not survive (recall 0.89) than those who did (0.74). The drop from cross-validation to the test set is moderate. With only 179 test passengers some variation is expected, but it's a sign that the tuned model fits the training folds slightly too closely.

---

## ⚠️ Limitations

- Small dataset (891 passengers), so the scores change noticeably with a different random split.
- `Cabin` was dropped (77% missing), although the deck letter could carry useful information.

---

## 🚀 Next Steps

- [ ] Try the deck letter from `Cabin` and ticket-group size as features
- [ ] Add a learning curve to check over- and under-fitting
- [ ] Submit predictions to the Kaggle competition and report the leaderboard score

---

## 📁 Repository Contents

```
Titanic-analysis/
├── data/
│   └── titanic.csv           # Dataset
├── analysis.ipynb            # Full analysis: EDA, modelling, evaluation
├── Titanic_report.html       # Automated profiling report (data-profiling)
├── deployment/
│   ├── main.py               # FastAPI app (feature engineering + prediction endpoint)
│   └── titanic_model.pkl     # Trained pipeline (preprocessing + tuned Gradient Boosting)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ▶️ How to Run

```bash
git clone https://github.com/MaykInvest/Titanic-analysis.git
cd Titanic-analysis

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

jupyter lab analysis.ipynb
```

---

## 🔮 Using the Saved Model

The pipeline handles missing values, scaling and encoding. The engineered features must be computed first:

```python
import joblib
import pandas as pd

model = joblib.load("deployment/titanic_model.pkl")

passenger = pd.DataFrame([{
    "pclass": 3, "sex": "female", "age": 22, "sibsp": 1, "parch": 0,
    "fare": 7.25, "embarked": "S", "title": "Mrs",
    "familysize": 2, "isalone": 0, "agegroup": "Adult",
}])

print(f"Survival probability: {model.predict_proba(passenger)[0, 1]:.1%}")
```

Load it with the same library versions used to train it (scikit-learn 1.7).

---

## 🌐 Prediction API (FastAPI)

The model is also served as a REST API. The API takes the raw passenger details (including the name) and does the feature engineering itself.

```bash
cd deployment
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/docs** for the interactive documentation, or call it directly:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"name": "Braund, Mr. Owen Harris", "pclass": 3, "sex": "male", "age": 22, "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "S"}'
```

Response:

```json
{"survived": false, "survival_probability": 0.078}
```

`age` and `embarked` are optional (missing values are filled the same way as during training). Invalid input, such as `"sex": "Male"` or a negative age, is rejected with a clear `422` error.

---

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · scikit-optimize · SHAP · FastAPI · data-profiling · Matplotlib · Seaborn · joblib

---

## 👤 Author

**Maykon Jonattan** — [GitHub](https://github.com/MaykInvest) · [LinkedIn](https://www.linkedin.com/in/maykon-oliveira-maykinvest/)
