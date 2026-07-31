# Test Data Files

This folder contains sample URL files for batch scanning demonstrations and testing.

## 📁 Files

### 1. `demo_urls.txt` (5 URLs)
**Quick demo** - Alternating legitimate and suspicious URLs
- Perfect for 30-second demonstrations
- Mix of clear legitimate and obvious phishing patterns

### 2. `test_urls.txt` (10 URLs)
**Standard test** - Basic validation set
- 5 legitimate URLs (Google, GitHub, Wikipedia, Amazon, Microsoft)
- 5 suspicious URLs (typosquatting, long subdomains, etc.)
- Plain text format, one URL per line

### 3. `test_urls.csv` (10 URLs)
**Test with descriptions** - Same as test_urls.txt but with context
- Columns: `url`, `description`
- Descriptions explain each URL type
- Good for presentations and reports

### 4. `batch_test_comprehensive.csv` (20 URLs)
**Full validation suite** - Comprehensive testing
- 10 legitimate URLs from varied categories
- 10 phishing URLs with different attack patterns
- Columns: `url`, `category`, `expected_result`
- Use for model validation and accuracy calculations

## 🎯 How to Use

### In the Application:

1. Run: `streamlit run app/main.py`
2. Navigate to **URL Scanner** page
3. Select **Batch Scan** mode
4. Choose **Upload File**
5. Select any file from this folder
6. Click **Scan URLs**
7. View results and download CSV report

### File Formats:

**Text files (.txt):**
```
https://example.com
https://test.com
```
One URL per line

**CSV files (.csv):**
```csv
url,description,category
https://example.com,Description,Category
```
First column must be `url`, other columns are optional and ignored by scanner

## 📊 For Reports & Demos

### Quick Demo Flow:
1. Use `demo_urls.txt` (30 seconds)
2. Show real-time scanning
3. Display results table
4. Download CSV results

### Full Validation:
1. Use `batch_test_comprehensive.csv`
2. Compare results with `expected_result` column
3. Calculate accuracy: (Correct Predictions / Total URLs) × 100%
4. Include in assignment report

## ⚠️ Notes

- All files use URLs from test set or public examples
- Private IP addresses (192.168.x.x) will be rejected
- CSV files: Only first column is scanned, others are for reference
- Results include timestamps in SGT (Singapore Time)
