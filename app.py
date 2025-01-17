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
    st.markdown('<p style="text-align: center; color: #718096; font-size: 0.95rem; margin-bottom: 2rem;">Powered by Advanced ML Algorithms | Interactive Analytics Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module:",
        ["🏠 Home", "🎾 Classification (Tennis)", "⛳ Regression (Golf)", "🔍 Clustering Analysis"]
    )
    
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
    st.markdown('<div class="sub-header">Project Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎾 Classification
        **Dataset:** Australian Open Tennis  
        **Objective:** Predict match outcomes  
        **Algorithms:** 5 models
        - Logistic Regression
        - K-Nearest Neighbors
        - Gaussian Naive Bayes
        - Decision Tree
        - Support Vector Machine
        """)
    
    with col2:
        st.markdown("""
        ### ⛳ Regression
        **Dataset:** PGA Golf 2022  
        **Objective:** Predict scoring average  
        **Algorithms:** 10 models
        - Linear Regression
        - Ridge, Lasso, ElasticNet
        - Polynomial Regression
        - Decision Tree
        - Random Forest
        - Gradient Boosting
        - SVR, KNN
        """)
    
    with col3:
        st.markdown("""
        ### 🔍 Clustering
        **Dataset:** Tennis Matches  
        **Objective:** Group similar matches  
        **Techniques:**
        - K-Means Clustering
        - Elbow Method
        - Silhouette Analysis
        - PCA Visualization
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 📊 Features
    - **Interactive UI** - Easy-to-use Streamlit interface
    - **Real-time Training** - Train models on-demand
    - **Comprehensive Metrics** - Detailed performance analysis
    - **Visualizations** - Confusion matrices, ROC curves, feature importance
    - **Model Comparison** - Side-by-side algorithm evaluation
    """)


def show_classification():
    """Classification module - Australian Open Tennis"""
    st.markdown('<div class="sub-header">🎾 Tennis Match Classification</div>', unsafe_allow_html=True)
    st.write("Predict whether the higher-ranked player will win an Australian Open match")
    
    # Load data
    X, y, df = load_classification_data()
    
    if X is None:
        st.error("⚠️ Could not load classification data. Please ensure 'AustralianOpen.csv' is in the application directory.")
        return
    
    # Dataset info
    with st.expander("📊 Dataset Information"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Matches", len(X))
        col2.metric("Features", X.shape[1])
        col3.metric("Class Balance", f"{(y.mean() * 100):.1f}% Yes")
        
        st.write("**Feature Description:**")
        st.write("- `rank_gap_abs`: Absolute ranking difference between players")
        st.write("- `experience_gap_abs`: Absolute experience difference (years)")
        st.write("- `height_gap_abs`: Absolute height difference (cm)")
        st.write("- `weight_gap_abs`: Absolute weight difference (kg)")
        st.write("- `same_handedness_bin`: Whether both players have same playing hand (1=Yes, 0=No)")
    
    # Train models
    if st.button("🚀 Train All Classification Models", key="train_class"):
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
            st.success("✅ Training Complete!")
            
            results_df = pd.DataFrame(results).sort_values('F1 Score', ascending=False)
            
            st.markdown("### 📈 Model Performance Comparison")
            st.dataframe(results_df.style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1 Score': '{:.4f}',
                'ROC AUC': '{:.4f}'
            }).highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'], color='lightgreen'))
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']].plot(
                kind='bar', ax=ax
            )
            ax.set_title('Classification Model Comparison', fontsize=16, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Confusion matrices
            st.markdown("### 🎯 Confusion Matrices")
            cols = st.columns(3)
            for idx, (name, y_pred) in enumerate(predictions.items()):
                with cols[idx % 3]:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    cm = confusion_matrix(y_test, y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
                    disp.plot(ax=ax, cmap='Blues', values_format='d')
                    ax.set_title(f'{name}')
                    st.pyplot(fig)
            
            # ROC Curves
            st.markdown("### 📊 ROC Curves")
            fig, ax = plt.subplots(figsize=(10, 8))
            for name, y_prob in probabilities.items():
                RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
            ax.plot([0, 1], [0, 1], linestyle='--', label='Random baseline', color='gray')
            ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best model
            best_model = results_df.iloc[0]['Model']
            st.markdown(f"### 🏆 Best Model: **{best_model}**")
            st.write(f"**F1 Score:** {results_df.iloc[0]['F1 Score']:.4f}")


def show_regression():
    """Regression module - PGA Golf"""
    st.markdown('<div class="sub-header">⛳ Golf Score Regression</div>', unsafe_allow_html=True)
    st.write("Predict PGA golfer's average score based on driving and putting statistics")
    
    # Load data
    X, y, df = load_regression_data()
    
    if X is None:
        st.error("⚠️ Could not load regression data. Please ensure 'PGA2022.csv' is in the application directory.")
        return
    
    # Dataset info
    with st.expander("📊 Dataset Information"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Observations", len(X))
        col2.metric("Features", X.shape[1])
        col3.metric("Avg Score Mean", f"{y.mean():.2f}")
        col4.metric("Avg Score Std", f"{y.std():.2f}")
        
        st.write("**Feature Description:**")
        st.write("- `avgDriveDist`: Average driving distance (yards)")
        st.write("- `drivePct`: Driving accuracy percentage")
        st.write("- `avgPuttsPerRound`: Average putts per round")
        st.write("- `onePuttPct`: One-putt percentage")
        st.write("- `driveSG`: Strokes gained driving")
        st.write("- `puttsSG`: Strokes gained putting")
        st.write("- `totalSG`: Combined strokes gained (engineered feature)")
    
    # Train models
    if st.button("🚀 Train All Regression Models", key="train_reg"):
        with st.spinner("Training 10 regression models..."):
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
            for idx, (name, model) in enumerate(models.items()):
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
            
            # Display results
            st.success("✅ Training Complete!")
            
            results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)
            
            st.markdown("### 📈 Model Performance Comparison")
            st.dataframe(results_df.style.format({
                'R² Score': '{:.4f}',
                'RMSE': '{:.4f}',
                'MAE': '{:.4f}',
                'MSE': '{:.4f}'
            }).highlight_max(subset=['R² Score'], color='lightgreen').highlight_min(
                subset=['RMSE', 'MAE', 'MSE'], color='lightgreen'
            ))
            
            # Visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # R² Score
            results_df.plot(x='Model', y='R² Score', kind='bar', ax=axes[0, 0], legend=False, color='skyblue')
            axes[0, 0].set_title('R² Score (Higher is Better)', fontweight='bold')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # RMSE
            results_df.plot(x='Model', y='RMSE', kind='bar', ax=axes[0, 1], legend=False, color='salmon')
            axes[0, 1].set_title('RMSE (Lower is Better)', fontweight='bold')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # MAE
            results_df.plot(x='Model', y='MAE', kind='bar', ax=axes[1, 0], legend=False, color='lightgreen')
            axes[1, 0].set_title('MAE (Lower is Better)', fontweight='bold')
            axes[1, 0].set_ylabel('MAE')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # MSE
            results_df.plot(x='Model', y='MSE', kind='bar', ax=axes[1, 1], legend=False, color='plum')
            axes[1, 1].set_title('MSE (Lower is Better)', fontweight='bold')
            axes[1, 1].set_ylabel('MSE')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Predicted vs Actual for top 3 models
            st.markdown("### 🎯 Predicted vs Actual (Top 3 Models)")
            top_3_models = results_df.head(3)['Model'].tolist()
            
            cols = st.columns(3)
            for idx, model_name in enumerate(top_3_models):
                with cols[idx]:
                    y_pred = predictions[model_name]
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(y_test, y_pred, alpha=0.5)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax.set_xlabel('Actual Score')
                    ax.set_ylabel('Predicted Score')
                    ax.set_title(f'{model_name}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            # Residual plots for top 3 models
            st.markdown("### 📉 Residual Plots (Top 3 Models)")
            cols = st.columns(3)
            for idx, model_name in enumerate(top_3_models):
                with cols[idx]:
                    y_pred = predictions[model_name]
                    residuals = y_test - y_pred
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(y_pred, residuals, alpha=0.5)
                    ax.axhline(y=0, color='r', linestyle='--', lw=2)
                    ax.set_xlabel('Predicted Score')
                    ax.set_ylabel('Residuals')
                    ax.set_title(f'{model_name}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            # Best model
            best_model = results_df.iloc[0]['Model']
            st.markdown(f"### 🏆 Best Model: **{best_model}**")
            st.write(f"**R² Score:** {results_df.iloc[0]['R² Score']:.4f}")
            st.write(f"**RMSE:** {results_df.iloc[0]['RMSE']:.4f}")


def show_clustering():
    """Clustering module - Tennis Match Analysis"""
    st.markdown('<div class="sub-header">🔍 Tennis Match Clustering</div>', unsafe_allow_html=True)
    st.write("Group tennis matches based on match statistics using K-Means clustering")
    
    st.info("⚠️ Note: For clustering analysis, please ensure you have the WTA Grand Slam dataset available.")
    
    # Clustering parameters
    col1, col2 = st.columns(2)
    with col1:
        k_min = st.slider("Minimum K", 2, 5, 2)
    with col2:
        k_max = st.slider("Maximum K", 5, 10, 8)
    
    if st.button("🚀 Run Clustering Analysis", key="cluster"):
        st.info("🔄 Clustering analysis would run here with the WTA dataset. Please refer to the clustering notebook for detailed implementation.")
        
        # Show example visualization structure
        st.markdown("### 📊 Elbow Method")
        st.write("Plot showing optimal K selection")
        
        st.markdown("### 📈 Silhouette Scores")
        st.write("Silhouette score for each K value")
        
        st.markdown("### 🎨 PCA Visualization")
        st.write("2D visualization of clusters using PCA")


if __name__ == "__main__":
    main()
