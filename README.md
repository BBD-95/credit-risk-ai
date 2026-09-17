# Credit Risk AI — Machine Learning Credit Risk Prediction

A personal data science project that predicts whether a loan applicant is likely to repay a loan, trained on the classic German Credit dataset. The project goes beyond a notebook: the trained model is served through a FastAPI REST API and exposed via a small trilingual web demo.

## 🎯 Objective

Given information about a loan applicant (age, income situation, requested amount, credit history, etc.), predict the probability that they will be a **good payer** or a **bad payer**, and expose that prediction through a usable interface — not just a Jupyter notebook.

## 📊 Dataset

* **Source**: [German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (UCI Machine Learning Repository)
* **Size**: 1,000 loan applicants
* **Features**: 20 explanatory variables (7 numerical, 13 categorical) — age, credit amount, duration, credit history, purpose, employment duration, checking account status, etc.
* **Target**: `credit\\\\\\\\\\\\\\\_risk`, binary (1 = good payer, 0 = bad payer)
* **Class balance**: 700 good payers / 300 bad payers (70% / 30%) — an imbalanced classification problem

## 🧹 Data preprocessing

* Exploratory data analysis (distribution plots for age, amount, duration, credit history vs. target)
* **One-Hot Encoding** on the 13 categorical columns (`pandas.get\\\\\\\\\\\\\\\_dummies`, `drop\\\\\\\\\\\\\\\_first=True`) 20 columns → 48 columns
* **Feature scaling** (`StandardScaler`) for the linear baseline model
* **Train/test split**: 80% / 20%, stratified on the target to preserve the 70/30 class balance in both sets

## 🤖 Models

Two models were trained and compared:

|Model|Notes|
|-|-|
|Logistic Regression|Baseline, `class\\\\\\\\\\\\\\\_weight='balanced'`, features scaled|
|**Random Forest** (final model)|`n\\\\\\\\\\\\\\\_estimators=200`, `max\\\\\\\\\\\\\\\_depth=10`, `class\\\\\\\\\\\\\\\_weight='balanced'`, no scaling needed|

Random Forest was selected as the final model — it outperformed the baseline on every metric.

## 📈 Evaluation

Test set: 200 applicants (60 bad payers / 140 good payers).

|Metric|Bad payer (0)|Good payer (1)|
|-|-|-|
|Precision|0.50|0.84|
|Recall|0.68|0.71|
|F1-score|0.58|0.77|

**Overall accuracy: 70%**

Accuracy alone is not a reliable metric on an imbalanced dataset like this one, precision/recall per class give a much more honest picture, which is why both are reported here rather than accuracy in isolation.

### Feature importance

The Random Forest's top predictors were, in order: checking account status, credit amount, loan duration, applicant age, and credit history. Checking account status turned out to be a stronger predictor than the loan amount or duration, a useful reminder to let the data challenge assumptions rather than the other way around.

## 🌐 Interactive web interface

A standalone HTML/CSS/JS demo (`frontend/dossier-credit.html`) that calls the API live and renders the prediction as a probability-based verdict.

* **🌍 Available in English, French, and Arabic** (with full right-to-left layout support for Arabic)
* Quick-fill demo profiles to test contrasting applicant scenarios without manual data entry
* No framework — a single self-contained HTML file

## 🔌 API (FastAPI)

* `GET /`  health check
* `POST /predict`  takes a JSON payload describing an applicant (validated with Pydantic) and returns the prediction, the predicted class, and both class probabilities
* Automatic interactive documentation available at `/docs` (Swagger UI)
* CORS enabled for local use with the static frontend

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/BBD-95/credit-risk-ai.git
cd credit-risk-ai

# 2. Create and activate a virtual environment
python -m venv venv
venv\\\\\\\\\\\\\\\\Scripts\\\\\\\\\\\\\\\\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API (from the project root)
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`, and the interactive docs at `http://127.0.0.1:8000/docs`.

To use the demo interface, open `frontend/dossier-credit.html` directly in a browser while the API is running.

## 🗂️ Project structure

```
credit-risk-ai/
├── api/
│   └── main.py              # FastAPI application
├── data/
│   └── GermanCredit.csv     # Raw dataset
├── frontend/
│   └── dossier-credit.html  # Trilingual demo interface
├── models/
│   ├── model.pkl            # Trained Random Forest model
│   ├── scaler.pkl           # StandardScaler (used for the logistic regression baseline)
│   └── columns.pkl          # Encoded feature columns, in training order
├── notebooks/
│   └── 01\\\\\\\\\\\\\\\_exploration.ipynb # Data exploration, preprocessing, and model training
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧪 Example prediction

Request to `POST /predict`:

```json
{
  "status": "... < 100 DM",
  "duration": 48,
  "credit\\\\\\\\\\\\\\\_history": "existing credits paid back duly till now",
  "purpose": "domestic appliances",
  "amount": 5951,
  "savings": "... < 100 DM",
  "employment\\\\\\\\\\\\\\\_duration": "1 <= ... < 4 years",
  "installment\\\\\\\\\\\\\\\_rate": 2,
  "personal\\\\\\\\\\\\\\\_status\\\\\\\\\\\\\\\_sex": "female : divorced/separated/married",
  "other\\\\\\\\\\\\\\\_debtors": "none",
  "present\\\\\\\\\\\\\\\_residence": 2,
  "property": "real estate",
  "age": 22,
  "other\\\\\\\\\\\\\\\_installment\\\\\\\\\\\\\\\_plans": "none",
  "housing": "own",
  "number\\\\\\\\\\\\\\\_credits": 1,
  "job": "skilled employee/official",
  "people\\\\\\\\\\\\\\\_liable": 1,
  "telephone": "no",
  "foreign\\\\\\\\\\\\\\\_worker": "yes"
}
```

Response:

```json
{
  "prediction": 0,
  "resultat": "Mauvais payeur",
  "probabilite\\\\\\\\\\\\\\\_bon\\\\\\\\\\\\\\\_payeur": 0.143,
  "probabilite\\\\\\\\\\\\\\\_mauvais\\\\\\\\\\\\\\\_payeur": 0.857
}
```

## 🔮 Future improvements

* Try gradient boosting models (XGBoost / LightGBM) for comparison
* Hyperparameter tuning with cross-validation (GridSearchCV / RandomizedSearchCV)
* Address class imbalance with resampling techniques (SMOTE) in addition to `class\\\\\\\\\\\\\\\_weight`
* Add automated tests for the API
* Containerize the API with Docker for easier deployment

## 📝 Note

This is a personal learning project built to practice the full data science workflow, from raw data to a deployed, usable interface  not a production-grade credit scoring system.

## 👤 Author

Feel free to reach out with questions or feedback.

Ben Dhaou 

Belgacem

www.linkedin.com/in/ben-dhaou-belgacem-650925259





