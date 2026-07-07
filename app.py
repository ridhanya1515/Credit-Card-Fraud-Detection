"""
Credit Card Fraud Detection - Streamlit Web Application

Interactive dashboard for real-time fraud detection using Kaggle dataset.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import FraudDetectionPredictor
from src.config import MODELS_DIR, KAGGLE_DATASET_PATH, V_FEATURES
from src.utils import load_kaggle_data

# =================== PAGE CONFIGURATION ===================
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================== CUSTOM THEME & STYLING ===================
st.markdown("""
    <style>
    /* Alert Boxes */
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .danger-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .info-box {
        background-color: #e7f3ff;
        border: 2px solid #2196F3;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    
    /* Main Title */
    .main-title {
        text-align: center;
        color: #1f4e79;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    /* Sidebar Styling */
    .sidebar-header {
        font-size: 18px;
        font-weight: bold;
        color: #1f4e79;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# =================== CACHING ===================
@st.cache_resource
def load_predictor():
    """Load model and cache it."""
    try:
        return FraudDetectionPredictor(model_dir=MODELS_DIR)
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    """Load dataset and cache it."""
    try:
        return load_kaggle_data()
    except FileNotFoundError:
        return None

def initialize_session_state():
    """Initialize session state variables."""
    if 'predictor' not in st.session_state:
        st.session_state.predictor = load_predictor()
    if 'df' not in st.session_state:
        st.session_state.df = load_data()
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False

# =================== UTILITY FUNCTIONS ===================
def get_sample_transaction(class_type="legitimate"):
    """Get realistic transaction from dataset."""
    df = st.session_state.df
    if df is None:
        return None
    
    class_val = 0 if class_type == "legitimate" else 1
    subset = df[df["Class"] == class_val].sample(1)  # Random sample
    feature_cols = V_FEATURES + ["Amount", "Time"]
    return subset[feature_cols].values[0]

def get_random_transaction():
    """Get random transaction from dataset."""
    df = st.session_state.df
    if df is None:
        return None
    
    subset = df.sample(1, random_state=None)
    feature_cols = V_FEATURES + ["Amount", "Time"]
    return subset[feature_cols].values[0]

# =================== SIDEBAR ===================
def render_sidebar():
    """Render sidebar navigation."""
    with st.sidebar:
        # Sidebar Header
        st.markdown("<div class='sidebar-header'>🗺️ Navigation</div>", unsafe_allow_html=True)
        
        page = st.radio(
            "Navigate to:",
            ["🏠 Prediction", "📊 Dashboard", "🔍 How It Works", "📈 Model Info", "📋 Dataset"],
            key="page",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("<div class='sidebar-header'>📌 Dataset Overview</div>", unsafe_allow_html=True)
        
        if st.session_state.df is not None:
            df = st.session_state.df
            total = len(df)
            fraud = (df["Class"] == 1).sum()
            fraud_pct = fraud / total * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Txns", f"{total:,}")
                st.metric("Fraud %", f"{fraud_pct:.2f}%")
            with col2:
                st.metric("Fraud Cases", f"{fraud:,}")
                st.metric("Legit %", f"{100-fraud_pct:.2f}%")
        
        st.markdown("---")
        st.markdown("<div class='sidebar-header'>⚙️ Quick Info</div>", unsafe_allow_html=True)
        st.info("""
        **🤖 Model:** Gradient Boosting
        
        **📊 Dataset:** Kaggle
        
        **🎯 Accuracy:** ~98%
        
        **⏱️ Inference:** <100ms
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 12px; color: gray;'>
            <p>📧 <strong>Built with</strong></p>
            <p>Python • Pandas • Scikit-learn • Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
        
        return page

# =================== PREDICTION PAGE ===================
def render_prediction_page():
    """Render prediction page."""
    predictor = st.session_state.predictor
    
    if predictor is None:
        st.error("❌ Model not found. Run: `python train.py`")
        return
    
    st.markdown("## 🎯 Fraud Detection")
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Legitimate Sample", use_container_width=True):
            st.session_state.test_features = get_sample_transaction("legitimate")
            st.session_state.sample_type = "Legitimate (Real)"
            st.rerun()
    
    with col2:
        if st.button("⚠️ Fraud Sample", use_container_width=True):
            st.session_state.test_features = get_sample_transaction("fraud")
            st.session_state.sample_type = "Fraud (Real)"
            st.rerun()
    
    with col3:
        if st.button("🎲 Random", use_container_width=True):
            st.session_state.test_features = get_random_transaction()
            st.session_state.sample_type = "Random"
            st.rerun()
    
    st.markdown("---")
    
    # CSV upload
    st.markdown("### Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            feature_cols = V_FEATURES + ["Amount", "Time"]
            
            if all(col in df_upload.columns for col in feature_cols):
                if st.button("Batch Predict", use_container_width=True):
                    results = predictor.predict_batch(df_upload[feature_cols].values)
                    df_upload['prediction'] = [r['prediction'] for r in results]
                    df_upload['confidence'] = [r['confidence'] for r in results]
                    
                    st.dataframe(
                        df_upload[['Amount', 'Time', 'prediction', 'confidence']],
                        use_container_width=True
                    )
                    
                    fraud_count = sum(1 for r in results if r['is_fraud'])
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total", len(results))
                    col2.metric("Fraud", fraud_count)
                    col3.metric("Legitimate", len(results) - fraud_count)
            else:
                st.error(f"Missing columns: {', '.join(feature_cols)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.markdown("### Single Prediction")
    
    # Get features
    if hasattr(st.session_state, 'test_features'):
        st.info(f"Using: {st.session_state.sample_type}")
        features = st.session_state.test_features.copy()
    else:
        features = np.zeros(30)
    
    # Input in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### PCA Features (V1-V14)")
        for i in range(14):
            features[i] = st.number_input(
                f"V{i+1}", value=float(features[i]), step=0.1, key=f"v{i+1}"
            )
    
    with col2:
        st.markdown("#### PCA Features (V15-V28)")
        for i in range(14, 28):
            features[i] = st.number_input(
                f"V{i+1}", value=float(features[i]), step=0.1, key=f"v{i+1}"
            )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        features[28] = st.number_input("Amount ($)", value=float(features[28]), min_value=0.0, step=0.01)
    with col2:
        features[29] = st.number_input("Time (sec)", value=float(features[29]), min_value=0.0, step=1.0)
    
    # Predict
    if st.button("🔍 Predict", use_container_width=True, key="predict"):
        try:
            result = predictor.predict(features)
            st.session_state.last_result = result
            st.session_state.prediction_made = True
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Display result
    if st.session_state.prediction_made:
        render_result(st.session_state.last_result)

def render_result(result):
    """Render prediction result."""
    st.markdown("---")
    st.markdown("## 📊 Result")
    
    is_fraud = result['is_fraud']
    confidence = result['confidence']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_text = "🔴 FRAUD" if is_fraud else "✅ LEGITIMATE"
        color = "#dc3545" if is_fraud else "#28a745"
        st.markdown(f"<h3 style='color: {color}'>{pred_text}</h3>", unsafe_allow_html=True)
    
    with col2:
        st.metric("Confidence", f"{confidence*100:.1f}%")
    
    with col3:
        risk = "🔴 HIGH" if is_fraud else "🟢 LOW"
        st.metric("Risk", risk)
    
    # Probability chart
    fig = go.Figure(data=[
        go.Bar(
            y=['Fraud', 'Legitimate'],
            x=[confidence*100, (1-confidence)*100],
            orientation='h',
            marker_color=['#dc3545', '#28a745'],
            text=[f'{confidence*100:.1f}%', f'{(1-confidence)*100:.1f}%'],
            textposition='auto'
        )
    ])
    fig.update_layout(showlegend=False, height=200, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # Alert
    if is_fraud:
        st.markdown(
            """<div class='danger-box'>
            <h4>⚠️ Fraudulent Transaction</h4>
            <p>This transaction is flagged as fraud. Further verification recommended.</p>
            </div>""", unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div class='success-box'>
            <h4>✅ Legitimate Transaction</h4>
            <p>This transaction appears legitimate.</p>
            </div>""", unsafe_allow_html=True
        )
    
    st.caption(f"Predicted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =================== DASHBOARD PAGE ===================
def render_dashboard_page():
    """Render dashboard."""
    df = st.session_state.df
    
    if df is None:
        st.error("❌ Dataset not loaded. Download creditcard.csv and place in data/")
        return
    
    st.markdown("## 📊 Dataset Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    fraud = (df["Class"] == 1).sum()
    legit = (df["Class"] == 0).sum()
    
    col1.metric("Total", f"{total:,}")
    col2.metric("Fraud", f"{fraud:,}")
    col3.metric("Legitimate", f"{legit:,}")
    col4.metric("Fraud %", f"{fraud/total*100:.3f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Class Distribution")
        fig = px.pie(
            values=[legit, fraud],
            names=['Legitimate', 'Fraud'],
            color_discrete_map={'Legitimate': '#28a745', 'Fraud': '#dc3545'},
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Amount Distribution")
        df_filt = df[df['Amount'] < df['Amount'].quantile(0.95)]
        fig = px.histogram(
            df_filt, x='Amount', nbins=50, color='Class',
            color_discrete_map={0: '#28a745', 1: '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Feature Correlation")
    top_cols = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "Amount", "Class"]
    corr = df[top_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmid=0))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# =================== HOW IT WORKS PAGE ===================
def render_how_it_works_page():
    """Render how it works."""
    st.markdown("## 🔍 How It Works")
    
    with st.expander("📊 Dataset", expanded=True):
        st.markdown("""
        **Kaggle Credit Card Fraud Detection**
        - 284,807 transactions | 492 fraud (0.17%)
        - Features: V1-V28 (PCA), Amount, Time, Class
        """)
    
    with st.expander("🛠️ Preprocessing"):
        st.markdown("""
        1. **Train/Test Split** (80/20) - before scaling
        2. **Scaling** - StandardScaler on Amount & Time only
        3. **Imbalance** - SMOTE + Undersampling on training data
        """)
    
    with st.expander("🤖 Models"):
        st.markdown("""
        - Logistic Regression (baseline)
        - Random Forest (ensemble)
        - Gradient Boosting ⭐ (best)
        """)
    
    with st.expander("🔮 Prediction"):
        st.markdown("""
        1. Input 30 features
        2. Scale Amount & Time
        3. Model prediction
        4. Get probability
        5. Classify as fraud/legitimate
        """)

# =================== MODEL INFO PAGE ===================
def render_model_info_page():
    """Render model info."""
    st.markdown("## 🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dataset")
        st.info("""
        - Transactions: 284,807
        - Fraud: 492 (0.17%)
        - Legitimate: 284,315
        - Features: 30
        """)
    
    with col2:
        st.markdown("### Best Model")
        st.info("""
        - Algorithm: Gradient Boosting
        - Estimators: 150
        - Learning Rate: 0.08
        - Max Depth: 4
        """)
    
    st.markdown("---")
    st.markdown("### Performance")
    st.dataframe(
        pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            "Score": ["~98%", "~84%", "~80%", "~82%", "~0.95"]
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("**Note:** Run `python train.py` for actual metrics")

# =================== DATASET PAGE ===================
def render_dataset_page():
    """Render dataset info."""
    df = st.session_state.df
    
    if df is None:
        st.error("""
        ❌ Dataset not found
        
        1. Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
        2. Extract creditcard.csv to data/
        3. Refresh page
        """)
        return
    
    st.markdown("## 📋 Dataset")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    st.markdown("---")
    st.markdown("### Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("### Info")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[col].dtype) for col in df.columns],
        "Missing": [df[col].isnull().sum() for col in df.columns]
    })
    st.dataframe(info_df, use_container_width=True)

# =================== MAIN ===================
def main():
    """Main app."""
    initialize_session_state()
    
    # Header Banner
    st.markdown("""
        <div style='text-align: center; padding: 30px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; margin-bottom: 20px;'>
            <h1 style='color: white; margin: 0;'>💳 Credit Card Fraud Detection System</h1>
            <p style='color: rgba(255,255,255,0.9); margin: 10px 0 0 0;'>
                Real-time ML-based Fraud Detection using Kaggle Dataset
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    page = render_sidebar()
    
    try:
        if page == "🏠 Prediction":
            render_prediction_page()
        elif page == "📊 Dashboard":
            render_dashboard_page()
        elif page == "🔍 How It Works":
            render_how_it_works_page()
        elif page == "📈 Model Info":
            render_model_info_page()
        else:
            render_dataset_page()
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.caption("💳 Credit Card Fraud Detection | ML Portfolio | [GitHub](https://github.com/ridhanya1515/Credit-Card-Fraud-Detection)")
        st.caption("Built with ❤️ | Dataset: Kaggle | Model: Gradient Boosting"
)

if __name__ == "__main__":
    main()
