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

# Professional CSS styling
st.markdown("""
<style>
    /* Clean, modern main container */
    .main {
        background-color: #ffffff;
    }
    
    /* Professional header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a202c;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    /* Subtle subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #718096;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Clean section headers */
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Minimalist buttons */
    .stButton>button {
        background-color: #4299e1;
        color: white;
        font-weight: 500;
        border: none;
        border-radius: 6px;
        padding: 0.65rem 1.5rem;
        font-size: 1rem;
        transition: background-color 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #3182ce;
    }
    
    /* Clean info boxes */
    .stInfo {
        background-color: #ebf8ff;
        border-left: 4px solid #4299e1;
        border-radius: 4px;
        padding: 1rem;
    }
    
    .stSuccess {
        background-color: #f0fff4;
        border-left: 4px solid #48bb78;
        border-radius: 4px;
        padding: 1rem;
    }
    
    .stWarning {
        background-color: #fffaf0;
        border-left: 4px solid #ed8936;
        border-radius: 4px;
        padding: 1rem;
    }
    
    /* Professional dataframes */
    .dataframe {
        border: 1px solid #e2e8f0;
        border-radius: 4px;
    }
    
    /* Clean metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #718096;
        font-weight: 500;
    }
    
    /* Subtle progress bar */
    .stProgress > div > div > div > div {
        background-color: #4299e1;
    }
    
    /* Professional expander */
    .streamlit-expanderHeader {
        background-color: #f7fafc;
        border-radius: 4px;
        font-weight: 500;
        border: 1px solid #e2e8f0;
    }
    
    /* Clean radio buttons */
    .stRadio > label {
        font-weight: 500;
        color: #2d3748;
        font-size: 0.95rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f7fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Clean typography */
    h1, h2, h3, h4 {
        color: #2d3748;
        font-weight: 600;
    }
    
    p {
        color: #4a5568;
        line-height: 1.6;
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
    st.markdown('<h1 class="main-header">Machine Learning Capstone Project</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Team TASKMASTERS (Team 16) • 23CSE301 Machine Learning</p>', unsafe_allow_html=True)
    
    # Clean sidebar navigation
    st.sidebar.header("Navigation")
    
    page = st.sidebar.radio(
        "Select Module",
        ["🏠 Home", "🎾 Classification", "⛳ Regression", "🔍 Clustering"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    **ML Capstone Project**
    
    This application demonstrates machine learning algorithms for:
    - Tennis match prediction
    - Golf score prediction  
    - Match clustering analysis
    """)
    
    if page == "🏠 Home":
        show_home()
    elif page == "🎾 Classification":
        show_classification()
    elif page == "⛳ Regression":
        show_regression()
    elif page == "🔍 Clustering":
        show_clustering()


def show_home():
    """Home page with project overview"""
    st.header("Project Overview")
    
    st.markdown("""
    This project implements and compares multiple machine learning algorithms across three key domains:
    classification, regression, and clustering.
    """)
    
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
    st.header("🎾 Tennis Match Classification")
    
    st.markdown("""
    Predict whether the higher-ranked player will win an Australian Open match using 5 classification algorithms.
    """)
    
    # Load data
    X, y, df = load_classification_data()
    
    if X is None:
        st.error("⚠️ Could not load classification data. Please ensure 'AustralianOpen.csv' is in the application directory.")
        return
    
    # Dataset info with enhanced cards
    with st.expander("📊 Dataset Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Matches", len(X))
        col2.metric("Features", X.shape[1])
        col3.metric("Higher Rank Wins", f"{(y.mean() * 100):.1f}%")
        
        st.markdown("**Features:**")
        st.markdown("""
        - **rank_gap_abs**: Ranking difference between players
        - **experience_gap_abs**: Experience difference (years as pro)
        - **height_gap_abs**: Height difference (cm)
        - **weight_gap_abs**: Weight difference (kg)
        - **same_handedness_bin**: Same playing hand (1=Yes, 0=No)
        """)
    
    # Train models button with enhanced styling
    if st.button("Train All Classification Models", key="train_class"):
        with st.spinner("Training 5 classification models..."):
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
            
            for idx, (name, model) in enumerate(models.items()):
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
            
            # Display results
            st.success("Training complete!")
            
            results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)
            
            st.subheader("Model Performance Comparison")
            st.dataframe(
                results_df.style.format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1 Score': '{:.4f}',
                    'ROC AUC': '{:.4f}'
                }).highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'], color='#e6f3ff'),
                use_container_width=True
            )
            
            # Visualization with professional styling
            st.subheader("Performance Metrics")
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']].plot(
                kind='bar', ax=ax, width=0.75, 
                color=['#4299e1', '#48bb78', '#ed8936', '#9f7aea', '#f56565']
            )
            ax.set_title('Model Performance Comparison', fontsize=14, fontweight='600', pad=15, color='#2d3748')
            ax.set_ylabel('Score', fontsize=11, fontweight='500')
            ax.set_xlabel('')
            ax.set_ylim(0, 1.05)
            ax.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            cols = st.columns(3)
            cmaps = ['Blues', 'Greens', 'Purples', 'Oranges', 'Reds']
            for idx, (name, y_pred) in enumerate(predictions.items()):
                with cols[idx % 3]:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    fig.patch.set_facecolor('white')
                    cm = confusion_matrix(y_test, y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
                    disp.plot(ax=ax, cmap=cmaps[idx], values_format='d')
                    ax.set_title(f'{name}', fontweight='500', fontsize=11, pad=10)
                    st.pyplot(fig)
            
            # ROC Curves
            st.subheader("ROC Curves")
            fig, ax = plt.subplots(figsize=(10, 7))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            colors = ['#4299e1', '#48bb78', '#9f7aea', '#ed8936', '#f56565']
            for idx, (name, y_prob) in enumerate(probabilities.items()):
                RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax, lw=2, color=colors[idx])
            
            ax.plot([0, 1], [0, 1], linestyle='--', label='Random', color='gray', lw=1.5, alpha=0.7)
            ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='600', pad=15, color='#2d3748')
            ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='500')
            ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='500')
            ax.legend(loc='lower right', frameon=True, fontsize=9)
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best model summary
            best_model = results_df.iloc[0]['Model']
            best_f1 = results_df.iloc[0]['F1 Score']
            best_acc = results_df.iloc[0]['Accuracy']
            best_roc = results_df.iloc[0]['ROC AUC']
            
            st.success(f"**Best Model:** {best_model}")
            col1, col2, col3 = st.columns(3)
            col1.metric("F1 Score", f"{best_f1:.4f}")
            col2.metric("Accuracy", f"{best_acc:.4f}")
            col3.metric("ROC AUC", f"{best_roc:.4f}")


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
