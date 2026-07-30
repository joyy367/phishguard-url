# PhishGuard URL

URL-only phishing detection web application for EATC Assignment 2 (2026).

The app extracts 22 lexical and structural features from URL text. It does not
visit the destination, download page content, or execute JavaScript.

## Local setup

Use Python 3.11 for both development and deployment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes Jupyter, Matplotlib, and Seaborn.
Streamlit Community Cloud installs the smaller production set from
`requirements.txt`.

Open the notebooks with:

```powershell
jupyter lab notebooks
```

Notebook roles:

1. `01_eda.ipynb` explores the raw dataset and class distribution.
2. `02_preprocessing.ipynb` demonstrates production cleaning and the
   domain-exclusive split.
3. `03_model_training.ipynb` is a reporting-only evaluation notebook. Run the
   training script first, then run this notebook to display the saved model
   comparison, final metrics, curves, feature importance, and safe error
   analysis. It does not fit models or generate test predictions.

## Dataset

Download **URL-Phish v2** from Mendeley Data
(DOI `10.17632/65z9twcx3r.2`) and place it at:

```text
data/Dataset.csv
```

The training pipeline treats `url` and `label` as the authoritative columns.
It normalises every URL and recomputes all 22 numerical features through the
same extractor used by the deployed application.

## Train the models

```powershell
python scripts/train_models.py --data data/Dataset.csv
```

The pipeline:

1. Validates binary labels and checks conflicts before deduplication.
2. Recomputes the complete feature contract from URL text.
3. Creates domain-exclusive 70/15/15 train, validation, and test subsets.
4. Compares five candidates using validation phishing F1.
5. Computes feature importance on validation data.
6. Refits the winner on train plus validation.
7. Evaluates once on the domain-unseen test set.

After this command finishes, run `03_model_training.ipynb` from top to bottom
and save its outputs for the assignment evidence.

Important generated artifacts include:

```text
models/best_model.pkl
models/best_model_name.pkl
models/feature_names.pkl
models/best_model_evaluation.json
models/model_comparison.csv
reports/validation_comparison.csv
reports/final_test_metrics.csv
reports/permutation_importance.csv
reports/test_analysis.csv
reports/test_curves.json
reports/recommended_demo_cases_defanged.csv
```

`reports/full_test_set_results.csv` and
`reports/recommended_demo_cases.csv` contain raw URLs and are intentionally
gitignored. Keep them local.

## Run and test

```powershell
python -m pytest -q
python scripts/validate_deployment.py
streamlit run app/main.py
```

The pre-deployment validator must finish with zero failures before deployment.

## Streamlit Community Cloud

1. Push the source, notebooks, generated model artifacts, and safe report
   artifacts to GitHub.
2. Do not commit `data/`, the SQLite database, secrets, or raw URL result files.
3. Set the entrypoint to `app/main.py`.
4. Select Python 3.11 in Advanced settings.
5. Confirm all four pages and the PDF download in the deployed application.

Scan history is isolated by browser session and stored in an ephemeral local
SQLite file. Cloud restarts may clear it.

## Dataset citation

Dam Minh, Linh; Tran Cong, Hung (2026), “URL-Phish: A Feature-Engineered
Dataset for Phishing Detection”, Mendeley Data, Version 2,
doi: `10.17632/65z9twcx3r.2` (CC BY 4.0).
