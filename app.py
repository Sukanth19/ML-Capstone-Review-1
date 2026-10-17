"""
ML Capstone Project - Full-Fledged Application with UI
Team TASKMASTERS (Team 16)
23CSE301 Machine Learning

This application provides a comprehensive interface for:
- Classification (5 algorithms): Australian Open Tennis Match Prediction
- Regression (10 algorithms): PGA Golf Score Prediction
- Clustering: Tennis Match Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, RocCurveDisplay,
    r2_score, mean_squared_error, mean_absolute_error, silhouette_score
)

# Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC

# Regression algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Clustering algorithms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
RANDOM_STATE = 42

# Page configuration
st.set_page_config(
    page_title="ML Capstone Project",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced visual appeal
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling with gradient */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        letter-spacing: -1px;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #4a5568;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Sub-header with gradient border */
    .sub-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #f093fb;
    }
    
    /* Enhanced metric cards with shadows */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.6);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    /* Styled buttons with gradient */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        transform: translateY(-2px);
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes with gradient borders */
    .stInfo {
        background-color: #e6f3ff;
        border-left: 5px solid #667eea;
        border-radius: 8px;
    }
    
    .stSuccess {
        background-color: #e6ffe6;
        border-left: 5px solid #4caf50;
        border-radius: 8px;
    }
    
    .stError {
        background-color: #ffe6e6;
        border-left: 5px solid #f44336;
        border-radius: 8px;
    }
    
    /* Enhanced dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Card containers for modules */
    .module-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        margin: 1rem 0;
        border: 2px solid transparent;
        background-clip: padding-box;
        position: relative;
    }
    
    .module-card::before {
        content: '';
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
        z-index: -1;
        margin: -2px;
        border-radius: inherit;
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Metric value styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Radio button styling */
    .stRadio > label {
        font-weight: 600;
        color: #2d3748;
    }
</style>
""", unsafe_allow_html=True)


# Utility Functions
@st.cache_data
def load_classification_data():
    """Load and prepare classification dataset"""
    try:
        df = pd.read_csv('AustralianOpen.csv', encoding='latin1')
        
        # Feature engineering
        working_df = df.copy()
        working_df['rank_gap_abs'] = (working_df['WRank'] - working_df['LRank']).abs()
        working_df['experience_gap_abs'] = (working_df['pl1_year_pro'] - working_df['pl2_year_pro']).abs()
        working_df['height_gap_abs'] = (working_df['pl1_height'] - working_df['pl2_height']).abs()
        working_df['weight_gap_abs'] = (working_df['pl1_weight'] - working_df['pl2_weight']).abs()
        working_df['same_handedness_bin'] = working_df['same_handedness'].astype('string').str.strip().str.title().map({'Yes': 1, 'No': 0})
        working_df['target'] = working_df['higher_rank_won'].astype('string').str.strip().str.title().map({'No': 0, 'Yes': 1})
        
        feature_columns = ['rank_gap_abs', 'experience_gap_abs', 'height_gap_abs', 'weight_gap_abs', 'same_handedness_bin']
        model_df = working_df[feature_columns + ['target']].dropna(subset=['target']).copy()
        
        X = model_df[feature_columns]
        y = model_df['target'].astype(int)
        
        return X, y, working_df
    except Exception as e:
        st.error(f"Error loading classification data: {e}")
        return None, None, None


@st.cache_data
def load_regression_data():
    """Load and prepare regression dataset"""
    try:
        df = pd.read_csv('PGA2022.csv', encoding='latin1')
        
        # Feature engineering
        working_df = df.copy()
        working_df['totalSG'] = working_df['driveSG'] + working_df['puttsSG']
        
        feature_columns = ['avgDriveDist', 'drivePct', 'avgPuttsPerRound', 'onePuttPct', 'driveSG', 'puttsSG', 'totalSG']
        model_df = working_df[feature_columns + ['avgScore']].dropna()
        
        X = model_df[feature_columns]
        y = model_df['avgScore']
        
        return X, y, working_df
    except Exception as e:
        st.error(f"Error loading regression data: {e}")
        return None, None, None


def get_classification_models():
    """Get 5 classification algorithms"""
    scaled_preprocessor = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), slice(None))
    ])
    
    tree_preprocessor = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), slice(None))
    ])
    
    models = {
        'Logistic Regression': Pipeline([
            ('preprocess', scaled_preprocessor),
            ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        ]),
        'K-Nearest Neighbors': Pipeline([
            ('preprocess', scaled_preprocessor),
            ('model', KNeighborsClassifier(n_neighbors=5))
        ]),
        'Gaussian Naive Bayes': Pipeline([
            ('preprocess', scaled_preprocessor),
            ('model', GaussianNB())
        ]),
        'Decision Tree': Pipeline([
            ('preprocess', tree_preprocessor),
            ('model', DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5))
        ]),
        'Support Vector Machine': Pipeline([
            ('preprocess', scaled_preprocessor),
            ('model', SVC(probability=True, random_state=RANDOM_STATE, kernel='rbf'))
        ])
    }
    
    return models


def get_regression_models():
    """Get 10 regression algorithms"""
    scaled_preprocessor = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), slice(None))
    ])
    
    tree_preprocessor = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), slice(None))
    ])
    
    models = {
        '1. Linear Regression': Pipeline([
            ('preprocess', scaled_preprocessor),
            ('model', LinearRegression())
        ]),
        '2. Ridge Regression': Pipeline([
            ('preprocess', scaled_preprocessor),
            ('model', Ridge(alpha=1.0, random_state=RANDOM_STATE))
        ]),
        '3. Lasso Regression': Pipeline([
            ('preprocess', scaled_preprocessor),
            ('model', Lasso(alpha=0.1, random_state=RANDOM_STATE))
        ]),
        '4. ElasticNet': Pipeline([
            ('preprocess', scaled_preprocessor),
            ('model', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE))
        ]),
        '5. Polynomial Regression': Pipeline([
            ('preprocess', scaled_preprocessor),
            ('poly', PolynomialFeatures(degree=2)),
            ('model', LinearRegression())
        ]),
        '6. Decision Tree': Pipeline([
            ('preprocess', tree_preprocessor),
            ('model', DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=5))
        ]),
        '7. Random Forest': Pipeline([
            ('preprocess', tree_preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE))
        ]),
        '8. Gradient Boosting': Pipeline([
            ('preprocess', tree_preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE))
        ]),
        '9. Support Vector Regressor': Pipeline([
            ('preprocess', scaled_preprocessor),
            ('model', SVR(kernel='rbf'))
        ]),
        '10. K-Nearest Neighbors': Pipeline([
            ('preprocess', scaled_preprocessor),
            ('model', KNeighborsRegressor(n_neighbors=5))
        ])
    }
    
    return models


# Main Application
def main():
    st.markdown('<h1 class="main-header">🎓 Machine Learning Capstone Project 🚀</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">✨ Team TASKMASTERS (Team 16) | 23CSE301 Machine Learning ✨</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='color: #718096; font-size: 0.95rem; margin: 0;'>
        🔬 Powered by Advanced ML Algorithms | 📊 Interactive Analytics Dashboard
        </p>
        <p style='color: #a0aec0; font-size: 0.85rem; margin: 0.25rem 0 0 0;'>
        Featuring Classification • Regression • Clustering Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar navigation with styling
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                   font-size: 2rem; margin: 0;'>🎯 Navigation</h1>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select Module:",
        ["🏠 Home", "🎾 Classification (Tennis)", "⛳ Regression (Golf)", "🔍 Clustering Analysis"]
    )
    
    # Add sidebar footer with info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-top: 2rem;'>
        <h4 style='color: white; margin: 0;'>💡 Quick Info</h4>
        <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem; margin-top: 0.5rem;'>
        This app demonstrates ML algorithms for classification, regression, and clustering tasks.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if page == "🏠 Home":
        show_home()
    elif page == "🎾 Classification (Tennis)":
        show_classification()
    elif page == "⛳ Regression (Golf)":
        show_regression()
    elif page == "🔍 Clustering Analysis":
        show_clustering()


def show_home():
    """Home page with project overview"""
    st.markdown('<div class="sub-header">📊 Project Overview</div>', unsafe_allow_html=True)
    
    # Add welcome message with gradient box
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 20px; color: white; text-align: center; 
                margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
        <h2 style='margin: 0; font-weight: 700;'>Welcome to ML Capstone Dashboard! 🚀</h2>
        <p style='margin-top: 1rem; font-size: 1.1rem; opacity: 0.95;'>
        Explore powerful machine learning algorithms through interactive visualizations and real-time analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='module-card'>
        <h3 style='color: #667eea;'>🎾 Classification</h3>
        <p><strong>Dataset:</strong> Australian Open Tennis</p>
        <p><strong>Objective:</strong> Predict match outcomes</p>
        <p><strong>Algorithms:</strong> 5 models</p>
        <ul style='color: #4a5568;'>
        <li>✓ Logistic Regression</li>
        <li>✓ K-Nearest Neighbors</li>
        <li>✓ Gaussian Naive Bayes</li>
        <li>✓ Decision Tree</li>
        <li>✓ Support Vector Machine</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='module-card'>
        <h3 style='color: #f093fb;'>⛳ Regression</h3>
        <p><strong>Dataset:</strong> PGA Golf 2022</p>
        <p><strong>Objective:</strong> Predict scoring average</p>
        <p><strong>Algorithms:</strong> 10 models</p>
        <ul style='color: #4a5568;'>
        <li>✓ Linear Regression</li>
        <li>✓ Ridge, Lasso, ElasticNet</li>
        <li>✓ Polynomial Regression</li>
        <li>✓ Decision Tree</li>
        <li>✓ Random Forest, Gradient Boosting</li>
        <li>✓ SVR, KNN</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='module-card'>
        <h3 style='color: #764ba2;'>🔍 Clustering</h3>
        <p><strong>Dataset:</strong> Tennis Matches</p>
        <p><strong>Objective:</strong> Group similar matches</p>
        <p><strong>Techniques:</strong></p>
        <ul style='color: #4a5568;'>
        <li>✓ K-Means Clustering</li>
        <li>✓ Elbow Method</li>
        <li>✓ Silhouette Analysis</li>
        <li>✓ PCA Visualization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Enhanced Features Section with icons
    st.markdown('<div class="sub-header">✨ Key Features</div>', unsafe_allow_html=True)
    
    feat_col1, feat_col2 = st.columns(2)
    
    with feat_col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 0.5rem 0; 
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);'>
            <h4 style='color: #667eea; margin-top: 0;'>🎨 Interactive UI</h4>
            <p style='color: #4a5568; margin-bottom: 0;'>Beautiful, easy-to-use Streamlit interface with modern design</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 0.5rem 0; 
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);'>
            <h4 style='color: #f093fb; margin-top: 0;'>⚡ Real-time Training</h4>
            <p style='color: #4a5568; margin-bottom: 0;'>Train multiple models on-demand with live progress tracking</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 0.5rem 0; 
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);'>
            <h4 style='color: #764ba2; margin-top: 0;'>📈 Comprehensive Metrics</h4>
            <p style='color: #4a5568; margin-bottom: 0;'>Detailed performance analysis with multiple evaluation metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 0.5rem 0; 
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);'>
            <h4 style='color: #667eea; margin-top: 0;'>📊 Rich Visualizations</h4>
            <p style='color: #4a5568; margin-bottom: 0;'>Confusion matrices, ROC curves, and feature importance plots</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 0.5rem 0; 
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);'>
            <h4 style='color: #f093fb; margin-top: 0;'>🔍 Model Comparison</h4>
            <p style='color: #4a5568; margin-bottom: 0;'>Side-by-side algorithm evaluation and benchmarking</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%); 
                    padding: 1.5rem; border-radius: 15px; margin: 0.5rem 0; 
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);'>
            <h4 style='color: #764ba2; margin-top: 0;'>💾 Export Results</h4>
            <p style='color: #4a5568; margin-bottom: 0;'>Download predictions and performance reports</p>
        </div>
        """, unsafe_allow_html=True)


def show_classification():
    """Classification module - Australian Open Tennis"""
    st.markdown('<div class="sub-header">🎾 Tennis Match Classification</div>', unsafe_allow_html=True)
    
    # Add attractive intro banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 20%, #764ba2 80%); 
                padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1.5rem;
                box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);'>
        <h3 style='margin: 0; color: white;'>🏆 Predict Australian Open Match Winners</h3>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.95;'>
        Using 5 powerful classification algorithms to predict whether the higher-ranked player will win
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    X, y, df = load_classification_data()
    
    if X is None:
        st.error("⚠️ Could not load classification data. Please ensure 'AustralianOpen.csv' is in the application directory.")
        return
    
    # Dataset info with enhanced cards
    with st.expander("📊 Dataset Information", expanded=False):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h4 style='margin: 0; color: #2d3748;'>📈 Dataset Statistics</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Matches", len(X), delta="Ready for analysis")
        col2.metric("Features", X.shape[1], delta="Engineered")
        col3.metric("Class Balance", f"{(y.mean() * 100):.1f}% Yes", delta="Higher rank wins")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #667eea;'>
            <h4 style='color: #667eea; margin-top: 0;'>🏆 Feature Description</h4>
            <ul style='color: #4a5568;'>
                <li><strong>rank_gap_abs:</strong> Absolute ranking difference between players</li>
                <li><strong>experience_gap_abs:</strong> Absolute experience difference (years as pro)</li>
                <li><strong>height_gap_abs:</strong> Absolute height difference (cm)</li>
                <li><strong>weight_gap_abs:</strong> Absolute weight difference (kg)</li>
                <li><strong>same_handedness_bin:</strong> Whether both players have same playing hand (1=Yes, 0=No)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Train models button with enhanced styling
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Train All Classification Models", key="train_class"):
        with st.spinner("🔄 Training 5 classification models... Please wait"):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
            )
            
            # Get models
            models = get_classification_models()
            
            # Train and evaluate
            results = []
            predictions = {}
            probabilities = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (name, model) in enumerate(models.items()):
                status_text.markdown(f"**Training:** {name}... ({idx + 1}/{len(models)})")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                predictions[name] = y_pred
                probabilities[name] = y_prob
                
                results.append({
                    'Model': name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'ROC AUC': roc_auc_score(y_test, y_prob)
                })
                
                progress_bar.progress((idx + 1) / len(models))
            
            status_text.empty()
            
            # Display results with enhanced success message
            st.markdown("""
            <div style='background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; 
                        box-shadow: 0 5px 15px rgba(132, 250, 176, 0.3); margin: 1rem 0;'>
                <h3 style='color: white; margin: 0;'>✅ Training Complete!</h3>
                <p style='color: white; margin: 0.5rem 0 0 0; opacity: 0.95;'>
                All 5 models trained successfully. Review the results below.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0;'>
                <h3 style='color: white; margin: 0;'>📈 Model Performance Comparison</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                results_df.style.format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1 Score': '{:.4f}',
                    'ROC AUC': '{:.4f}'
                }).highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'], color='lightgreen'),
                use_container_width=True
            )
            
            # Visualization with enhanced styling
            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(14, 7))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#ffffff')
            
            results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']].plot(
                kind='bar', ax=ax, width=0.8, 
                color=['#667eea', '#f093fb', '#764ba2', '#f5576c', '#19547b']
            )
            ax.set_title('Classification Model Comparison - All Metrics', 
                        fontsize=18, fontweight='bold', pad=20, color='#2d3748')
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_xlabel('Model', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left', 
                     frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Confusion matrices with enhanced header
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0;'>
                <h3 style='color: white; margin: 0;'>🎯 Confusion Matrices</h3>
            </div>
            """, unsafe_allow_html=True)
            cols = st.columns(3)
            cmaps = ['Blues', 'Purples', 'Greens', 'Oranges', 'Reds']
            for idx, (name, y_pred) in enumerate(predictions.items()):
                with cols[idx % 3]:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    fig.patch.set_facecolor('#f8f9fa')
                    cm = confusion_matrix(y_test, y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
                    disp.plot(ax=ax, cmap=cmaps[idx], values_format='d')
                    ax.set_title(f'{name}', fontweight='bold', fontsize=12, pad=10)
                    st.pyplot(fig)
            
            # ROC Curves with enhanced header
            st.markdown("""
            <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0;'>
                <h3 style='color: #2d3748; margin: 0;'>📊 ROC Curves - Model Comparison</h3>
            </div>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 9))
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#ffffff')
            
            colors = ['#667eea', '#f093fb', '#764ba2', '#f5576c', '#19547b']
            for idx, (name, y_prob) in enumerate(probabilities.items()):
                RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax, color=colors[idx], lw=2.5)
            
            ax.plot([0, 1], [0, 1], linestyle='--', label='Random baseline', color='gray', lw=2)
            ax.set_title('ROC Curves - All Classification Models', fontsize=16, fontweight='bold', pad=20, color='#2d3748')
            ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best model with trophy box
            best_model = results_df.iloc[0]['Model']
            best_f1 = results_df.iloc[0]['F1 Score']
            best_acc = results_df.iloc[0]['Accuracy']
            best_roc = results_df.iloc[0]['ROC AUC']
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%); 
                        padding: 2rem; border-radius: 20px; text-align: center; 
                        box-shadow: 0 10px 30px rgba(255, 216, 155, 0.4); margin: 2rem 0;'>
                <h2 style='color: white; margin: 0; font-size: 2.5rem;'>🏆 Best Model</h2>
                <h3 style='color: white; margin: 1rem 0;'>{best_model}</h3>
                <div style='display: flex; justify-content: space-around; margin-top: 1.5rem;'>
                    <div>
                        <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;'>F1 Score</p>
                        <p style='color: white; margin: 0; font-size: 1.8rem; font-weight: bold;'>{best_f1:.4f}</p>
                    </div>
                    <div>
                        <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;'>Accuracy</p>
                        <p style='color: white; margin: 0; font-size: 1.8rem; font-weight: bold;'>{best_acc:.4f}</p>
                    </div>
                    <div>
                        <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;'>ROC AUC</p>
                        <p style='color: white; margin: 0; font-size: 1.8rem; font-weight: bold;'>{best_roc:.4f}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def show_regression():
    """Regression module - PGA Golf"""
    st.markdown('<div class="sub-header">⛳ Golf Score Regression</div>', unsafe_allow_html=True)
    
    # Add attractive intro banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1.5rem;
                box-shadow: 0 8px 20px rgba(240, 147, 251, 0.3);'>
        <h3 style='margin: 0; color: white;'>⛳ PGA Golf Score Prediction</h3>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.95;'>
        Predict golfer's average score using 10 advanced regression algorithms based on driving and putting stats
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    X, y, df = load_regression_data()
    
    if X is None:
        st.error("⚠️ Could not load regression data. Please ensure 'PGA2022.csv' is in the application directory.")
        return
    
    # Dataset info with enhanced styling
    with st.expander("📊 Dataset Information", expanded=False):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                    padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h4 style='margin: 0; color: #2d3748;'>📈 Dataset Statistics</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Observations", len(X), delta="Golfers analyzed")
        col2.metric("Features", X.shape[1], delta="Performance metrics")
        col3.metric("Avg Score Mean", f"{y.mean():.2f}", delta="Strokes per round")
        col4.metric("Avg Score Std", f"{y.std():.2f}", delta="Score variation")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background: rgba(240, 147, 251, 0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #f093fb;'>
            <h4 style='color: #f093fb; margin-top: 0;'>⛳ Feature Description</h4>
            <ul style='color: #4a5568;'>
                <li><strong>avgDriveDist:</strong> Average driving distance (yards) - Power metric</li>
                <li><strong>drivePct:</strong> Driving accuracy percentage - Precision metric</li>
                <li><strong>avgPuttsPerRound:</strong> Average putts per round - Short game performance</li>
                <li><strong>onePuttPct:</strong> One-putt percentage - Putting efficiency</li>
                <li><strong>driveSG:</strong> Strokes gained driving - Performance vs field average</li>
                <li><strong>puttsSG:</strong> Strokes gained putting - Putting advantage</li>
                <li><strong>totalSG:</strong> Combined strokes gained (engineered feature) - Overall advantage</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Train models with enhanced styling
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Train All Regression Models", key="train_reg"):
        with st.spinner("🔄 Training 10 regression models... This may take a moment"):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=RANDOM_STATE
            )
            
            # Get models
            models = get_regression_models()
            
            # Train and evaluate
            results = []
            predictions = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (name, model) in enumerate(models.items()):
                status_text.markdown(f"**Training:** {name}... ({idx + 1}/{len(models)})")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                predictions[name] = y_pred
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'Model': name,
                    'R² Score': r2,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MSE': mse
                })
                
                progress_bar.progress((idx + 1) / len(models))
            
            status_text.empty()
            
            # Display results with enhanced success message
            st.markdown("""
            <div style='background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); 
                        padding: 1.5rem; border-radius: 15px; text-align: center; 
                        box-shadow: 0 5px 15px rgba(132, 250, 176, 0.3); margin: 1rem 0;'>
                <h3 style='color: white; margin: 0;'>✅ Training Complete!</h3>
                <p style='color: white; margin: 0.5rem 0 0 0; opacity: 0.95;'>
                All 10 regression models trained successfully. Analyzing performance metrics...
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
            
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0;'>
                <h3 style='color: white; margin: 0;'>📈 Model Performance Comparison</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                results_df.style.format({
                    'R² Score': '{:.4f}',
                    'RMSE': '{:.4f}',
                    'MAE': '{:.4f}',
                    'MSE': '{:.4f}'
                }).highlight_max(subset=['R² Score'], color='lightgreen').highlight_min(
                    subset=['RMSE', 'MAE', 'MSE'], color='lightgreen'
                ),
                use_container_width=True
            )
            
            # Visualization with enhanced styling and custom colors
            st.markdown("<br>", unsafe_allow_html=True)
            fig, axes = plt.subplots(2, 2, figsize=(16, 13))
            fig.patch.set_facecolor('#f8f9fa')
            
            # R² Score
            ax = axes[0, 0]
            ax.set_facecolor('#ffffff')
            results_df.plot(x='Model', y='R² Score', kind='bar', ax=ax, legend=False, color='#667eea', width=0.7)
            ax.set_title('R² Score (Higher is Better)', fontweight='bold', fontsize=14, pad=15, color='#2d3748')
            ax.set_ylabel('R² Score', fontweight='bold', fontsize=11)
            ax.set_xlabel('')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.tick_params(axis='x', rotation=45)
            
            # RMSE
            ax = axes[0, 1]
            ax.set_facecolor('#ffffff')
            results_df.plot(x='Model', y='RMSE', kind='bar', ax=ax, legend=False, color='#f093fb', width=0.7)
            ax.set_title('RMSE (Lower is Better)', fontweight='bold', fontsize=14, pad=15, color='#2d3748')
            ax.set_ylabel('RMSE', fontweight='bold', fontsize=11)
            ax.set_xlabel('')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.tick_params(axis='x', rotation=45)
            
            # MAE
            ax = axes[1, 0]
            ax.set_facecolor('#ffffff')
            results_df.plot(x='Model', y='MAE', kind='bar', ax=ax, legend=False, color='#f5576c', width=0.7)
            ax.set_title('MAE (Lower is Better)', fontweight='bold', fontsize=14, pad=15, color='#2d3748')
            ax.set_ylabel('MAE', fontweight='bold', fontsize=11)
            ax.set_xlabel('')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.tick_params(axis='x', rotation=45)
            
            # MSE
            ax = axes[1, 1]
            ax.set_facecolor('#ffffff')
            results_df.plot(x='Model', y='MSE', kind='bar', ax=ax, legend=False, color='#764ba2', width=0.7)
            ax.set_title('MSE (Lower is Better)', fontweight='bold', fontsize=14, pad=15, color='#2d3748')
            ax.set_ylabel('MSE', fontweight='bold', fontsize=11)
            ax.set_xlabel('')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Predicted vs Actual for top 3 models with enhanced header
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0;'>
                <h3 style='color: white; margin: 0;'>🎯 Predicted vs Actual (Top 3 Models)</h3>
            </div>
            """, unsafe_allow_html=True)
            top_3_models = results_df.head(3)['Model'].tolist()
            
            cols = st.columns(3)
            scatter_colors = ['#667eea', '#f093fb', '#f5576c']
            for idx, model_name in enumerate(top_3_models):
                with cols[idx]:
                    y_pred = predictions[model_name]
                    
                    fig, ax = plt.subplots(figsize=(7, 7))
                    fig.patch.set_facecolor('#f8f9fa')
                    ax.set_facecolor('#ffffff')
                    
                    ax.scatter(y_test, y_pred, alpha=0.6, color=scatter_colors[idx], s=60, edgecolors='white', linewidth=0.5)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2.5, label='Perfect Fit')
                    ax.set_xlabel('Actual Score', fontweight='bold', fontsize=11)
                    ax.set_ylabel('Predicted Score', fontweight='bold', fontsize=11)
                    ax.set_title(f'{model_name}', fontweight='bold', fontsize=12, pad=15)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.legend(loc='upper left')
                    st.pyplot(fig)
            
            # Residual plots for top 3 models with enhanced header
            st.markdown("""
            <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        padding: 1rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0;'>
                <h3 style='color: #2d3748; margin: 0;'>📉 Residual Plots (Top 3 Models)</h3>
            </div>
            """, unsafe_allow_html=True)
            cols = st.columns(3)
            residual_colors = ['#764ba2', '#19547b', '#f5576c']
            for idx, model_name in enumerate(top_3_models):
                with cols[idx]:
                    y_pred = predictions[model_name]
                    residuals = y_test - y_pred
                    
                    fig, ax = plt.subplots(figsize=(7, 7))
                    fig.patch.set_facecolor('#f8f9fa')
                    ax.set_facecolor('#ffffff')
                    
                    ax.scatter(y_pred, residuals, alpha=0.6, color=residual_colors[idx], s=60, edgecolors='white', linewidth=0.5)
                    ax.axhline(y=0, color='#e74c3c', linestyle='--', lw=2.5, label='Zero Residual')
                    ax.set_xlabel('Predicted Score', fontweight='bold', fontsize=11)
                    ax.set_ylabel('Residuals', fontweight='bold', fontsize=11)
                    ax.set_title(f'{model_name}', fontweight='bold', fontsize=12, pad=15)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.legend(loc='upper left')
                    st.pyplot(fig)
            
            # Best model with trophy box
            best_model = results_df.iloc[0]['Model']
            best_r2 = results_df.iloc[0]['R² Score']
            best_rmse = results_df.iloc[0]['RMSE']
            best_mae = results_df.iloc[0]['MAE']
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%); 
                        padding: 2rem; border-radius: 20px; text-align: center; 
                        box-shadow: 0 10px 30px rgba(255, 216, 155, 0.4); margin: 2rem 0;'>
                <h2 style='color: white; margin: 0; font-size: 2.5rem;'>🏆 Best Model</h2>
                <h3 style='color: white; margin: 1rem 0;'>{best_model}</h3>
                <div style='display: flex; justify-content: space-around; margin-top: 1.5rem;'>
                    <div>
                        <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;'>R² Score</p>
                        <p style='color: white; margin: 0; font-size: 1.8rem; font-weight: bold;'>{best_r2:.4f}</p>
                    </div>
                    <div>
                        <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;'>RMSE</p>
                        <p style='color: white; margin: 0; font-size: 1.8rem; font-weight: bold;'>{best_rmse:.4f}</p>
                    </div>
                    <div>
                        <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;'>MAE</p>
                        <p style='color: white; margin: 0; font-size: 1.8rem; font-weight: bold;'>{best_mae:.4f}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def show_clustering():
    """Clustering module - Tennis Match Analysis"""
    st.markdown('<div class="sub-header">🔍 Tennis Match Clustering</div>', unsafe_allow_html=True)
    
    # Add attractive intro banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                padding: 1.5rem; border-radius: 15px; color: #2d3748; margin-bottom: 1.5rem;
                box-shadow: 0 8px 20px rgba(168, 237, 234, 0.3);'>
        <h3 style='margin: 0; color: #2d3748;'>🔍 Unsupervised Match Clustering</h3>
        <p style='margin: 0.5rem 0 0 0; color: #4a5568;'>
        Group tennis matches based on match statistics using K-Means clustering with elbow method and PCA visualization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; 
                box-shadow: 0 5px 15px rgba(255, 236, 210, 0.4);'>
        <h4 style='color: #2d3748; margin: 0;'>ℹ️ Dataset Note</h4>
        <p style='color: #4a5568; margin: 0.5rem 0 0 0;'>
        For complete clustering analysis with K-Means, Elbow method, and PCA visualization, 
        please ensure you have the WTA Grand Slam dataset available. Refer to the clustering notebook for detailed implementation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clustering parameters in styled boxes
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0;'>
        <h4 style='color: white; margin: 0;'>⚙️ Clustering Parameters</h4>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        k_min = st.slider("Minimum K", 2, 5, 2)
    with col2:
        k_max = st.slider("Maximum K", 5, 10, 8)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Run Clustering Analysis", key="cluster"):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 1.5rem; border-radius: 15px; text-align: center; 
                    box-shadow: 0 5px 15px rgba(168, 237, 234, 0.3); margin: 1rem 0;'>
            <h4 style='color: #2d3748; margin: 0;'>🔄 Clustering Analysis Preview</h4>
            <p style='color: #4a5568; margin: 0.5rem 0 0 0;'>
            Clustering analysis would run here with the WTA dataset. Please refer to the clustering notebook for detailed implementation.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show example visualization structure with styled boxes
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0;'>
            <h4 style='color: white; margin: 0;'>📊 Elbow Method</h4>
        </div>
        """, unsafe_allow_html=True)
        st.write("📈 Plot showing optimal K selection based on within-cluster sum of squares")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0;'>
            <h4 style='color: white; margin: 0;'>📈 Silhouette Scores</h4>
        </div>
        """, unsafe_allow_html=True)
        st.write("📊 Silhouette score evaluation for each K value to measure cluster quality")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 1rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0;'>
            <h4 style='color: #2d3748; margin: 0;'>🎨 PCA Visualization</h4>
        </div>
        """, unsafe_allow_html=True)
        st.write("🔍 2D visualization of clusters using Principal Component Analysis")


if __name__ == "__main__":
    main()
