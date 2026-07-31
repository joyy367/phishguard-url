# PhishGuard URL

**URL-only phishing detection web application** for EATC Assignment 2 (2026).

This app extracts 22 lexical and structural features from URL text alone. It does **not** visit destinations, download page content, or execute JavaScript. A *Legitimate* prediction is not a safety guarantee.

---

## 🚀 Features

- **Single URL Scanner** - Analyze individual URLs with detailed results
- **Batch Scanner** - Upload .txt or .csv files to scan multiple URLs
- **Real-time Analysis** - 22 features extracted instantly
- **PDF Reports** - Download detailed assessment reports
- **Analytics Dashboard** - Model performance metrics and charts
- **Scan History** - Track all scans with local storage
- **Dark Theme UI** - Professional, modern interface

---

## 📋 Requirements

**Python Version:** 3.11 (required for deployment compatibility)

### Dependencies

All required packages are listed in:
- `requirements.txt` - Production dependencies (for Streamlit deployment)
- `requirements-dev.txt` - Development dependencies (includes Jupyter, testing tools)

---

## 🛠️ Local Setup

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

## 📊 Training the Model

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

## ▶️ Running the Application

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

## 🧪 Testing

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

## 📁 Batch Scanning Test Files

Ready-to-use test files are included in the `test_data/` folder:

### Available Test Files:

1. **`test_data/demo_urls.txt`** - Quick 5-URL demo (30 seconds)
2. **`test_data/test_urls.txt`** - Standard 10-URL test
3. **`test_data/test_urls.csv`** - 10 URLs with descriptions
4. **`test_data/batch_test_comprehensive.csv`** - Full 20-URL validation

### File Formats:

**Text files (.txt):** One URL per line
```
https://example.com
https://test.com
```

**CSV files (.csv):** First column = URLs (other columns ignored)
```csv
url,description,category
https://example.com,Description,Category
```

### How to Use:

1. Open app: `streamlit run app/main.py`
2. Go to **URL Scanner** → **Batch Scan**
3. Select **Upload File**
4. Choose any file from `test_data/` folder
5. Click **Scan URLs**
6. View results and download CSV

---

## 🎯 Demo URLs for Manual Testing

### ✅ Legitimate URLs:
```
https://www.google.com
https://github.com
https://en.wikipedia.org/wiki/Main_Page
https://www.amazon.com
https://www.microsoft.com
```

### ⚠️ Suspicious URLs:
```
https://paypal-verify-account-security-update.com
https://g00gle-login.com/verify
https://secure-bankofamerica-verify.com/login
https://login.security.update.verify.apple.com.verify-account.net
```

### 🚫 Blocked (Security Feature):
```
http://192.168.1.1/login
```
*Expected: Rejected - demonstrates private IP blocking*

---

## 📈 Model Performance

**Best Model:** Histogram Gradient Boosting Classifier

**Test Set Performance** (Domain-Unseen 15%):
- Accuracy: ~96.89%
- Phishing F1: ~89.48%
- Phishing Recall: ~92.21%
- ROC-AUC: ~0.9918
- PR-AUC: ~0.9639

*Note: Exact values depend on your training run*

---

## 🔧 Project Structure

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

## 🌐 Deployment (Streamlit Cloud)

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

## 🔒 Security Features

- **Private IP Blocking** - Rejects 192.168.x.x, 10.x.x.x, 127.x.x.x
- **URL Normalization** - Removes credentials and sensitive data
- **No Web Requests** - Never visits or downloads from URLs
- **Local Storage** - Scan history stored locally (SQLite)
- **Query String Removal** - Strips sensitive parameters before storage

---

## 📝 For Your Report

### Test Results Format:

| URL | Expected | Actual | Phishing Score | Risk Level |
|-----|----------|--------|----------------|------------|
| google.com | Legitimate | Legitimate | 1.59% | Low |
| paypal-verify... | Phishing | Phishing | 88.23% | High |

### Screenshots to Include:

1. Single URL scan results
2. Batch scan upload and results
3. Analytics dashboard
4. PDF report download
5. Feature importance chart
6. Model comparison table

### Key Metrics to Report:

- Model accuracy on test set
- Phishing F1 score
- False positive rate
- False negative rate
- Batch scan processing time
- Feature importance rankings

---

## 🐛 Known Issues

- **"undefined" text** may appear near gauge chart (cosmetic only, doesn't affect functionality)
- **Warnings** about feature names (suppressed in code, model works correctly)

---

## 📚 Feature Glossary

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

## 📧 Contact & Credits

**EATC Assignment 2 - 2026**

**Dataset:** URL-Phish v2 (DOI: 10.17632/65z9twcx3r.2)

**Technologies:**
- Python 3.11
- scikit-learn 1.3.2
- Streamlit 1.40.0
- Plotly 5.18.0
- ReportLab 4.1.0

---

## 📄 License

This project is for educational purposes (EATC Assignment 2).

---

## ✅ Quick Start Checklist

- [ ] Python 3.11 installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`requirements-dev.txt`)
- [ ] Dataset downloaded and placed in `data/Dataset.csv`
- [ ] Notebooks run in order (01 → 02 → 03)
- [ ] Model trained successfully
- [ ] Tests passing (`pytest`)
- [ ] App runs locally (`streamlit run app/main.py`)
- [ ] Batch test files work
- [ ] Ready for deployment!

---

**Last Updated:** January 2026
