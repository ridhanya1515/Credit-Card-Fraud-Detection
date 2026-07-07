# Screenshots Directory

This folder contains screenshots of the Credit Card Fraud Detection application for project documentation and portfolio display.

## Screenshot Guidelines

### 1_prediction_page.png
- **Location:** Home page of Streamlit app
- **Content:** Shows the fraud detection prediction interface
- **Includes:**
  - Header with project title
  - Three sample buttons (Legitimate, Fraud, Random)
  - CSV upload section
  - Single prediction form with input fields
  - Prediction result display with confidence bar
  - Success/warning boxes

**How to capture:**
```bash
streamlit run app.py
# Navigate to "Prediction" tab (default)
# Type some values in the form
# Click "Predict" button
# Take screenshot of the results section
```

---

### 02_dashboard.png
- **Location:** Dashboard tab in Streamlit app
- **Content:** Dataset statistics and visualizations
- **Includes:**
  - Metric cards (Total, Fraud, Legitimate, Fraud %)
  - Class distribution pie chart
  - Amount distribution histogram
  - Feature correlation heatmap
  - Interactive Plotly visualizations

**How to capture:**
```bash
streamlit run app.py
# Click on "📊 Dashboard" in sidebar
# Take screenshot showing all visualizations
```

---

### 03_model_info.png
- **Location:** Model Information tab
- **Content:** Model details and performance metrics
- **Includes:**
  - Dataset overview box
  - Model parameters box
  - Performance metrics table (Accuracy, Precision, Recall, F1, ROC-AUC)
  - Workflow diagram
  - Technical details

**How to capture:**
```bash
streamlit run app.py
# Click on "📈 Model Info" in sidebar
# Take screenshot of the page
```

---

### 04_dataset.png
- **Location:** Dataset tab
- **Content:** Dataset exploration and statistics
- **Includes:**
  - Dataset metrics (rows, columns, memory usage)
  - Raw data preview (first 10 rows)
  - Data types and missing values table
  - Statistical summary

**How to capture:**
```bash
streamlit run app.py
# Click on "📋 Dataset" in sidebar
# Take screenshot of the data preview section
```

---

## Adding Screenshots

1. **Run the application:**
   ```bash
   python train.py  # Train model first if not done
   streamlit run app.py
   ```

2. **Navigate to each page** and take high-quality screenshots

3. **Crop and optimize:**
   - Keep images at 1200px width for README display
   - Use PNG format for lossless quality
   - Name files clearly: `01_prediction_page.png`, etc.

4. **Place files in this directory** with exact names above

---

## Alternative: Using Browser DevTools

1. Open Streamlit app in browser
2. Press `F12` to open Developer Tools
3. Press `Ctrl+Shift+M` to enter Device Toolbar (responsive mode)
4. Set device to "Responsive" or "iPad" for good aspect ratio
5. Capture screenshots using Print Screen or browser's built-in screenshot tool

---

## Image Specifications

- **Format:** PNG preferred (lossless)
- **Width:** 1200-1400 pixels (for mobile-friendly display)
- **Quality:** High DPI (72+ DPI for web)
- **File size:** Try to keep under 500 KB each
- **Aspect ratio:** 16:9 or 4:3 preferred

---

## Privacy & Data

⚠️ **Important:** Ensure screenshots don't show:
- Real credit card numbers
- Personal user information
- Database credentials
- API keys

Use **sample data only** - the app can generate sample transactions from the dataset.

---

## Tools for Screenshot Editing

- **Snagit** - Professional screenshot tool
- **ShareX** - Free, open-source
- **Greenshot** - Lightweight Windows tool
- **GIMP** - Free image editor for post-processing
- **Online Tools:** Pixlr, Photopea

---

**Note:** These are placeholder references. Replace with actual screenshots from a running application for best portfolio presentation.
