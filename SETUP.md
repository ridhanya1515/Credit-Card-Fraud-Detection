## 🚀 Setup Instructions

### Step 1: Clone or Download the Project
```bash
git clone https://github.com/your-username/credit_card_fraud_detection.git
cd credit_card_fraud_detection
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Kaggle Dataset

#### Option A: Manual Download (Recommended for Beginners)
1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click "Download" button
3. Extract the ZIP file
4. Copy `creditcard.csv` to `data/creditcard.csv`

#### Option B: Using Kaggle CLI
1. Install Kaggle CLI: `pip install kaggle`
2. Setup Kaggle API key (see below)
3. Run: `kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ && unzip data/creditcardfraud.zip -d data/`

#### Setup Kaggle API Key
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/` (create folder if needed)
4. Run: `chmod 600 ~/.kaggle/kaggle.json` (macOS/Linux only)

### Step 5: Verify Dataset
Check if file exists: `data/creditcard.csv`

You should see:
```
data/
└── creditcard.csv  (284,807 rows, 30 columns)
```

### Step 6: Train the Model
```bash
python train.py
```

**Expected Output:**
```
======================================================================
   CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE
======================================================================

[1/6] Loading Kaggle Credit Card Fraud Detection Dataset...
   ✓ Total Transactions: 284,807
   ✓ Legitimate: 284,315 (99.83%)
   ✓ Fraudulent: 492 (0.17%)

[2/6] Exploratory Data Analysis...
   ✓ EDA plots saved to outputs/eda_analysis.png

[3/6] Data Preprocessing & Feature Engineering...
   Note: Data split BEFORE scaling to prevent data leakage
   ✓ Training set: 102,351 samples (resampled)
   ✓ Test set: 56,962 samples

[4/6] Training Models...
   → Training Logistic Regression...
   → Training Random Forest...
   → Training Gradient Boosting...

[5/6] Evaluating Models...
...

[6/6] Saving Best Model...
   ✓ Model saved to models/fraud_detection_model.joblib
   ✓ Scaler saved to models/feature_scaler.joblib
```

**Time Required:** 
- First run: ~10-15 minutes (depends on CPU)
- Subsequent runs: Cached if dataset unchanged

### Step 7: Run Web Application
```bash
streamlit run app.py
```

**Output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Then open http://localhost:8501 in your browser.

### Step 8: Make Predictions
```bash
python predict.py
```

---

## ✅ Verification Checklist

Run this to verify everything is set up correctly:

```bash
# Check Python version
python --version  # Should be 3.8+

# Check dependencies
pip list | grep -E "pandas|scikit-learn|streamlit"

# Check dataset
test -f data/creditcard.csv && echo "✓ Dataset found" || echo "✗ Dataset missing"

# Check models (after training)
test -f models/fraud_detection_model.joblib && echo "✓ Model found" || echo "✗ Model not trained yet - run: python train.py"

# Test imports
python -c "from src.utils import load_kaggle_data; print('✓ Imports OK')"
```

---

## 🐛 Common Issues & Solutions

### Issue: "Dataset not found at"
**Solution:**
```bash
# Verify file exists
ls -la data/creditcard.csv

# If missing, download from:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

### Issue: "ModuleNotFoundError: No module named 'X'"
**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or specific dependency
pip install pandas scikit-learn streamlit plotly
```

### Issue: "Model not found. Run python train.py first"
**Solution:**
```bash
# Train the model
python train.py

# This will create:
# - models/fraud_detection_model.joblib
# - models/feature_scaler.joblib
# - models/feature_columns.joblib
```

### Issue: Streamlit "Port 8501 already in use"
**Solution:**
```bash
streamlit run app.py --server.port 8502
# Or: Kill existing streamlit process
```

### Issue: Memory error during training  
**Solution:**
```bash
# Reduce batch size in config.py or:
# Use a machine with more RAM

# Alternatively, sample the dataset:
df = pd.read_csv('data/creditcard.csv').sample(frac=0.5, random_state=42)
df.to_csv('data/creditcard_small.csv', index=False)
```

### Issue: Slow training (> 30 minutes)
**Solution:**
```python
# In src/config.py, reduce:
"n_estimators": 100,  # Instead of 150
"learning_rate": 0.1,  # Instead of 0.08
```

---

## 📊 Project Workflow

```
1. Setup & Dataset
   └─ Clone → Install → Download Dataset

2. Training
   └─ python train.py
      ├─ Load dataset
      ├─ EDA
      ├─ Preprocess (no leakage)
      ├─ Train 3 models
      ├─ Evaluate
      └─ Save best model

3. Web Application
   └─ streamlit run app.py
      ├─ Prediction page (single & batch)
      ├─ Dashboard (statistics)
      ├─ How it works
      └─ Model info

4. Production
   └─ Deploy API, Docker, Cloud, etc.
```

---

## 🎯 Quick Commands

```bash
# Setup
git clone <repo> && cd credit_card_fraud_detection
python -m venv venv && source venv/bin/activate  # or use venv\Scripts\activate on Windows
pip install -r requirements.txt

# Train
python train.py

# Run app
streamlit run app.py

# Make predictions
python predict.py

# Check dataset
python -c "import pandas as pd; df = pd.read_csv('data/creditcard.csv'); print(f'Rows: {len(df)}, Fraud: {(df.Class==1).sum()}')"
```

---

## 📝 Notes

- ✅ Uses **real Kaggle dataset** (not synthetic)
- ✅ **No data leakage** (train/test split before preprocessing)
- ✅ **Class imbalance handled** (SMOTE + Undersampling)
- ✅ **Multiple models** (Logistic, RF, Gradient Boosting)
- ✅ **Comprehensive evaluation** (5+ metrics)
- ✅ **Interactive web app** (Streamlit)
- ✅ **Production-ready** (model persistence)
- ✅ **Best practices** (clean code, documentation)

---

**Next Steps:**
1. ✅ Download dataset
2. ✅ Install dependencies
3. ✅ Train model
4. ✅ Run web app
5. ✅ Make predictions
6. ✅ Push to GitHub
7. ✅ Add to portfolio/resume

Happy learning! 🚀
