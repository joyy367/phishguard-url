# PhishGuard URL

**URL-only phishing detection web application** for EATC Assignment 2 (2026).

This application uses machine learning to classify URLs as legitimate or phishing based on 22 lexical and structural features extracted from URL text alone. The system does not visit destination websites, download page content, or execute JavaScript, providing a safe and fast detection method.

---

## Overview

PhishGuard URL is an AI-powered cybersecurity application that addresses the growing threat of phishing attacks by analyzing URL patterns without requiring web requests. The system employs a Histogram Gradient Boosting classifier trained on approximately 11,000 URLs to achieve high accuracy in real-time phishing detection.

---

## Features

- **Single URL Scanner** - Analyze individual URLs with detailed results and confidence scores
- **Batch Scanner** - Upload .txt or .csv files to scan multiple URLs simultaneously
- **Real-time Analysis** - Extract and process 22 features instantly without web requests
- **PDF Reports** - Generate downloadable assessment reports with feature vectors
- **Analytics Dashboard** - View model performance metrics, ROC curves, and feature importance
- **Scan History** - Track all scans with session-isolated SQLite storage
- **Security Features** - Private IP blocking and URL normalization

---

## System Requirements

**Python Version:** 3.11 (required for deployment compatibility)

### Dependencies

All required packages are listed in:
- `requirements.txt` - Production dependencies (for Streamlit deployment)
- `requirements-dev.txt` - Development dependencies (includes Jupyter, testing tools)

---

## Installation and Setup

### 1. Create Virtual Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install Dependencies

**For development (includes notebooks and testing):**
```powershell
pip install -r requirements-dev.txt
```

**For production only:**
```powershell
pip install -r requirements.txt
```

### 3. Get the Dataset

Download **URL-Phish v2** from Mendeley Data (DOI: `10.17632/65z9twcx3r.2`)

Place it at:
```
data/Dataset.csv
```

The dataset should have columns: `url` and `label` (0 = legitimate, 1 = phishing)

---

## Model Training Pipeline

### Run the Notebooks in Order:

```powershell
jupyter lab notebooks
```

1. **`01_eda.ipynb`** - Exploratory Data Analysis
   - Dataset overview and statistics
   - Class distribution analysis
   - Feature correlation exploration

2. **`02_preprocessing.ipynb`** - Data Preprocessing
   - URL normalization
   - Feature extraction (22 features)
   - Domain-exclusive train/val/test split (70/15/15)
   - Creates: `models/train_val_test_split.pkl`

3. **`03_model_training.ipynb`** - Model Training & Evaluation
   - Trains 5 candidate models:
     - Logistic Regression
     - Random Forest
     - Histogram Gradient Boosting
     - XGBoost
     - Multi-layer Perceptron
   - Selects best model based on validation phishing F1
   - Evaluates on domain-unseen test set
   - Generates all reports and artifacts

### Key Training Details:

- **Split Method:** StratifiedGroupKFold (20 folds, grouped by domain)
  - 14 folds → training (70%)
  - 3 folds → validation (15%)
  - 3 folds → test (15%)
- **Selection Metric:** Phishing F1 (primary), Phishing Recall (tie-break)
- **Zero URL Overlap:** No URL appears in multiple splits
- **Zero Domain Overlap:** No domain appears in multiple splits
- **Feature Importance:** Calculated on validation set before retraining

### Generated Artifacts:

After training, these files are created:

**Models:**
```
models/best_model.pkl                    - Trained classifier
models/best_model_name.pkl               - Model name string
models/feature_names.pkl                 - Feature order (22 features)
models/best_model_evaluation.json        - Test set metrics
models/label_encoder.pkl                 - Label encoder (0/1 → Legitimate/Phishing)
```

**Reports:**
```
reports/validation_comparison.csv        - 5 models compared on validation set
reports/final_test_metrics.csv           - Final test performance
reports/permutation_importance.csv       - Feature importance rankings
reports/test_analysis.csv                - Detailed test results
reports/test_curves.json                 - ROC and PR curve data
reports/recommended_demo_cases_defanged.csv - Safe demo URLs from test set
```

**Visualizations:**
```
reports/fig_class_distribution.png
reports/fig_correlation_heatmap.png
reports/fig_feature_distributions.png
reports/fig_label_correlations.png
reports/fig_split_distribution.png
reports/fig_feature_importance.png
reports/fig_model_comparison.png
reports/fig_test_curves.png
```

---

## Running the Application

### Start the App:

```powershell
streamlit run app/main.py
```

The app will open at: `http://localhost:8501`

### Application Structure:

```
app/
├── main.py           - Streamlit UI and page routing
├── predictor.py      - Model loading and prediction
├── features.py       - Feature extraction (22 features)
├── database.py       - SQLite scan history storage
└── pdf_report.py     - PDF report generation
```

---

## Testing

### Run All Tests:

```powershell
python -m pytest -q
```

### Test Coverage:

- `tests/test_features.py` - Feature extraction validation
- `tests/test_predictor.py` - Model prediction pipeline
- `tests/test_database.py` - Scan history storage
- `tests/test_pdf_report.py` - PDF generation
- `tests/test_training_outputs.py` - Training artifacts validation

---

## Batch Scanning Test Files

Ready-to-use test files are included in the `test_data/` folder for demonstrations and validation:

### Available Test Files:

| File | URLs | Use Case | Time |
|------|------|----------|------|
| `demo_urls.txt` | 5 | Quick demo - alternating legitimate/phishing | 30 sec |
| `test_urls.txt` | 10 | Standard test - 5 legitimate, 5 phishing | 1 min |
| `test_urls.csv` | 10 | Same as above with descriptions | 1 min |
| `batch_test_comprehensive.csv` | 20 | Full validation - 10 legitimate, 10 phishing | 2 min |

### What's Included:

**Legitimate URLs (expected LOW phishing scores):**
- google.com, github.com, wikipedia.org, amazon.com, microsoft.com
- facebook.com, linkedin.com, stackoverflow.com, reddit.com, youtube.com

**Phishing URLs (expected HIGH phishing scores):**
- Typosquatting (g00gle, amaz0n, microsft)
- Many dashes (paypal-verify-account-security-update)
- Long subdomain chains (login.security.update.verify...)
- Urgency tactics (account-locked-verify-now)
- Banking impersonation (secure-bankofamerica-verify)

### File Formats:

**Text files (.txt):** One URL per line
```txt
https://www.google.com
https://github.com
https://paypal-verify-account-security-update.com
```

**CSV files (.csv):** First column must be URLs, other columns are optional and ignored
```csv
url,description,category,expected_result
https://www.google.com,Search engine,Legitimate,Pass
https://paypal-verify.com,Phishing site,Phishing,Fail
```

### Usage in Application:

1. Run: `streamlit run app/main.py`
2. Navigate to URL Scanner, then Batch Scan tab
3. Select Upload File option
4. Choose any file from test_data/ folder
5. Click Scan URLs button
6. View results table with predictions and confidence scores
7. Download results as CSV with Singapore timestamps

### Testing and Validation:

After batch scanning, the following metrics can be calculated:
- Accuracy: (Correct Predictions / Total URLs) × 100%
- False positive and false negative identification
- Comparison with expected results in CSV files
- Processing time and performance benchmarks

---

## Demo URLs for Manual Testing

### Legitimate URLs:
```
https://www.google.com
https://github.com
https://en.wikipedia.org/wiki/Main_Page
https://www.amazon.com
https://www.microsoft.com
```

### Suspicious URLs:
```
https://paypal-verify-account-security-update.com
https://g00gle-login.com/verify
https://secure-bankofamerica-verify.com/login
https://login.security.update.verify.apple.com.verify-account.net
```

### Blocked URLs (Security Feature Demonstration):
```
http://192.168.1.1/login
```
Note: Private IP addresses are rejected to prevent local network scanning.

---

## Model Performance

**Selected Model:** Histogram Gradient Boosting Classifier

**Test Set Performance** (Domain-Unseen 15% Split):
- Accuracy: ~96.89%
- Phishing F1 Score: ~89.48%
- Phishing Recall: ~92.21%
- Phishing Precision: ~86.91%
- ROC-AUC: ~0.9918
- PR-AUC: ~0.9639

Note: Values are approximate and may vary slightly depending on training run and random seed.

---

## Project Structure

```
PHISHGUARD_URL/
├── app/                              # Streamlit application
│   ├── main.py                       # Main UI and pages
│   ├── predictor.py                  # Model inference
│   ├── features.py                   # Feature extraction
│   ├── database.py                   # Scan history
│   └── pdf_report.py                 # PDF generation
├── data/
│   └── Dataset.csv                   # Training data (download separately)
├── models/                           # Trained model artifacts
│   ├── best_model.pkl
│   ├── best_model_name.pkl
│   ├── feature_names.pkl
│   ├── label_encoder.pkl
│   └── best_model_evaluation.json
├── notebooks/                        # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── reports/                          # Generated reports and charts
├── tests/                            # Unit tests
├── test_data/                        # Batch scanning test files
│   ├── demo_urls.txt                 # Quick 5-URL demo
│   ├── test_urls.txt                 # Standard 10-URL test
│   ├── test_urls.csv                 # 10 URLs with descriptions
│   └── batch_test_comprehensive.csv  # Full 20-URL validation
├── requirements.txt                  # Production dependencies
├── requirements-dev.txt              # Development dependencies
├── runtime.txt                       # Python version for deployment
└── README.md                         # This file
```

---

## Deployment (Streamlit Cloud)

### Prerequisites:
1. GitHub repository with your code
2. Streamlit Cloud account (sign in with GitHub)

### Steps:

1. **Push to GitHub:**
```powershell
git add .
git commit -m "Ready for deployment"
git push origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Select your repository: `joyy367/phishguard-url`
   - Branch: `main`
   - Main file path: `app/main.py`
   - Click "Deploy"

3. **Auto-Updates:**
   - Any push to `main` branch automatically redeploys
   - Wait 2-3 minutes for deployment
   - Refresh browser to see changes

### Deployment Files:
- `runtime.txt` - Specifies Python 3.11
- `requirements.txt` - Production dependencies only
- `.gitignore` - Excludes data files and local artifacts

---

## Security Features

- **Private IP Blocking** - Rejects 192.168.x.x, 10.x.x.x, 127.x.x.x
- **URL Normalization** - Removes credentials and sensitive data
- **No Web Requests** - Never visits or downloads from URLs
- **Local Storage** - Scan history stored locally (SQLite)
- **Query String Removal** - Strips sensitive parameters before storage

---

## Known Issues

- **Cosmetic Issue:** "undefined" text may appear near the gauge chart visualization. This is a display-only issue and does not affect functionality or prediction accuracy.
- **Console Warnings:** Feature name mismatch warnings from scikit-learn are suppressed in the code. The model functions correctly as feature order is validated on load.

---

## Feature Glossary

The model uses 22 URL-derived features:

1. `url_len` - Total URL character count
2. `dom_len` - Domain name length
3. `is_ip` - 1 if IP address, 0 otherwise
4. `tld_len` - Top-level domain length
5. `subdom_cnt` - Number of subdomains
6. `letter_cnt` - Alphabetic character count
7. `digit_cnt` - Numeric character count
8. `special_cnt` - Special character count
9. `eq_cnt` - Number of '=' characters
10. `qm_cnt` - Number of '?' characters
11. `amp_cnt` - Number of '&' characters
12. `dot_cnt` - Number of '.' characters
13. `dash_cnt` - Number of '-' characters
14. `under_cnt` - Number of '_' characters
15. `letter_ratio` - Letters / total length
16. `digit_ratio` - Digits / total length
17. `spec_ratio` - Special chars / total length
18. `is_https` - 1 if HTTPS, 0 if HTTP
19. `slash_cnt` - Number of '/' characters
20. `entropy` - Shannon entropy of URL string
21. `path_len` - Path component length
22. `query_len` - Query string length

---

## Credits and References

**Course:** Emerging AI Trends in Cybersecurity (EATC)  
**Assignment:** Assignment 2 (2026)  
**Institution:** Ngee Ann Polytechnic, School of Infocomm Technology  
**Programme:** Diploma in Cybersecurity & Digital Forensics

**Dataset Source:**  
URL-Phish v2 Dataset  
DOI: 10.17632/65z9twcx3r.2  
Mendeley Data

**Technology Stack:**
- Python 3.11
- scikit-learn 1.3.2 (Machine Learning)
- Streamlit 1.40.0 (Web Application Framework)
- Plotly 5.18.0 (Interactive Visualizations)
- ReportLab 4.1.0 (PDF Report Generation)
- SQLite (History Storage)

---

## License

This project is developed for educational purposes as part of EATC Assignment 2.

---
