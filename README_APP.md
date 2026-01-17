# ML Capstone Project - Full-Fledged Application with UI

## Team TASKMASTERS (Team 16) | 23CSE301 Machine Learning

This is a comprehensive machine learning application with an interactive web-based UI built using Streamlit.

## 🎯 Features

### 🎾 Classification Module (Tennis Match Prediction)
**Dataset:** Australian Open Tennis Matches (2008-2022)  
**Objective:** Predict whether the higher-ranked player will win

**5 Classification Algorithms:**
1. **Logistic Regression** - Probabilistic linear classifier
2. **K-Nearest Neighbors** - Instance-based learning
3. **Gaussian Naive Bayes** - Probabilistic classifier based on Bayes' theorem
4. **Decision Tree** - Tree-based hierarchical classifier
5. **Support Vector Machine** - Maximum margin classifier with RBF kernel

**Features:**
- Comprehensive performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Interactive confusion matrices for all models
- ROC curves comparison
- Model performance visualization
- Feature engineering (rank gaps, experience gaps, physical differences)

---

### ⛳ Regression Module (Golf Score Prediction)
**Dataset:** PGA Tour 2022 Statistics  
**Objective:** Predict golfer's average score

**10 Regression Algorithms:**
1. **Linear Regression** - Basic linear model
2. **Ridge Regression** - L2 regularization
3. **Lasso Regression** - L1 regularization (feature selection)
4. **ElasticNet** - Combined L1 + L2 regularization
5. **Polynomial Regression** - Non-linear polynomial features (degree 2)
6. **Decision Tree Regressor** - Tree-based non-linear model
7. **Random Forest Regressor** - Ensemble of 100 decision trees
8. **Gradient Boosting Regressor** - Sequential boosting ensemble
9. **Support Vector Regressor** - SVM for regression with RBF kernel
10. **K-Nearest Neighbors Regressor** - Instance-based regression

**Features:**
- Multiple regression metrics (R², RMSE, MAE, MSE)
- Predicted vs Actual scatter plots
- Residual analysis plots
- Model comparison visualizations
- Feature engineering (total strokes gained)

---

### 🔍 Clustering Module (Tennis Match Analysis)
**Dataset:** WTA Grand Slam Matches
**Objective:** Group similar matches based on statistics

**Techniques:**
- K-Means clustering
- Elbow method for optimal K selection
- Silhouette score analysis
- PCA visualization (2D projection)

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data Files

Ensure the following CSV files are in the same directory as `app.py`:
- `AustralianOpen.csv` - For classification module
- `PGA2022.csv` - For regression module

These files should already be in your project directory.

### Step 3: Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

---

## 📱 How to Use the Application

### Navigation
Use the sidebar to navigate between different modules:
- **🏠 Home** - Project overview and feature summary
- **🎾 Classification** - Tennis match prediction module
- **⛳ Regression** - Golf score prediction module
- **🔍 Clustering** - Tennis match clustering analysis

### Training Models

#### Classification Module:
1. Click on "🎾 Classification (Tennis)" in the sidebar
2. Expand "📊 Dataset Information" to view data details
3. Click "🚀 Train All Classification Models" button
4. Wait for training to complete (progress bar shows status)
5. View results:
   - Performance comparison table
   - Model comparison bar chart
   - Confusion matrices for all 5 models
   - ROC curves comparison
   - Best model identification

#### Regression Module:
1. Click on "⛳ Regression (Golf)" in the sidebar
2. Expand "📊 Dataset Information" to view data details
3. Click "🚀 Train All Regression Models" button
4. Wait for training to complete (progress bar shows status)
5. View results:
   - Performance comparison table
   - Four metric visualizations (R², RMSE, MAE, MSE)
   - Predicted vs Actual plots for top 3 models
   - Residual plots for top 3 models
   - Best model identification

---

## 📊 Model Performance Metrics

### Classification Metrics:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of actual positives correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve

### Regression Metrics:
- **R² Score**: Coefficient of determination (variance explained)
- **RMSE**: Root Mean Squared Error (in original units)
- **MAE**: Mean Absolute Error (average prediction error)
- **MSE**: Mean Squared Error (penalizes larger errors)

---

## 🎨 UI Features

### Interactive Elements:
- **Expandable sections** for dataset information
- **Progress bars** during model training
- **Color-coded metrics** (green highlighting for best values)
- **Interactive plots** (zoom, pan, save)
- **Responsive layout** adapts to screen size
- **Professional styling** with custom CSS

### Visualizations:
- Bar charts for model comparison
- Confusion matrices with color intensity
- ROC curves with multiple models
- Scatter plots for regression analysis
- Residual plots for error analysis

---

## 🏗️ Project Structure

```
ML-Capstone-Review-1/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README_APP.md                   # This file
│
├── AustralianOpen.csv             # Classification dataset
├── PGA2022.csv                    # Regression dataset
│
├── classification.ipynb           # Original classification notebook
├── regression.ipynb              # Original regression notebook
├── Team16Improvedclassification.ipynb
├── Team16_Clustering_Part1_KMeans.ipynb
│
└── README.md                      # Original project README
```

---

## 🔧 Technical Details

### Classification Implementation:
- **Data Preprocessing**: Median imputation + StandardScaler
- **Feature Engineering**: 
  - Absolute rank gap
  - Experience gap (years professional)
  - Height and weight differences
  - Handedness binary encoding
- **Train/Test Split**: 80/20 stratified split
- **Cross-validation**: Ready for 5-fold CV implementation

### Regression Implementation:
- **Data Preprocessing**: Median imputation + StandardScaler (where needed)
- **Feature Engineering**:
  - Combined strokes gained (totalSG = driveSG + puttsSG)
  - Driving and putting statistics
- **Train/Test Split**: 80/20 random split
- **Polynomial Features**: Degree 2 for polynomial regression

### Random State:
All models use `RANDOM_STATE = 42` for reproducibility.

---

## 💡 Tips for Best Results

1. **Internet Connection**: Required for first-time Streamlit setup
2. **Data Quality**: Ensure CSV files are not corrupted
3. **Browser**: Works best with Chrome, Firefox, or Edge
4. **Screen Size**: Optimal viewing on desktop/laptop (1920x1080+)
5. **Training Time**: Classification takes ~5-10 seconds, Regression takes ~10-15 seconds

---

## 🐛 Troubleshooting

### "Could not load data" error:
- Verify CSV files are in the same directory as `app.py`
- Check file names match exactly: `AustralianOpen.csv` and `PGA2022.csv`

### Slow performance:
- Close other resource-intensive applications
- Reduce number of data points if dataset is very large

### Import errors:
- Run `pip install -r requirements.txt` again
- Verify Python version is 3.8+

### Port already in use:
- Streamlit default port is 8501
- Use custom port: `streamlit run app.py --server.port 8502`

---

## 📝 Code Quality Features

- **Type hints** for better code readability
- **Docstrings** for all major functions
- **Error handling** with user-friendly messages
- **Caching** with `@st.cache_data` for performance
- **Modular design** with separate functions for each module
- **PEP 8** compliant code formatting

---

## 🎓 Academic Context

This application was developed as part of the 23CSE301 Machine Learning course capstone project. It demonstrates:

- **Supervised Learning**: Classification and Regression
- **Unsupervised Learning**: Clustering
- **Model Evaluation**: Multiple metrics and visualizations
- **Feature Engineering**: Domain-specific transformations
- **Software Engineering**: Production-ready web application

---

## 📚 References

- **Datasets**: SCORE Network (https://data.scorenetwork.org/)
- **Framework**: Streamlit (https://streamlit.io/)
- **ML Library**: Scikit-learn (https://scikit-learn.org/)

---

## 👥 Team TASKMASTERS

**Team 16 | 23CSE301 Machine Learning**

For questions or issues, please contact the team members or refer to the original notebooks for detailed methodology.

---

## 📄 License

This project is for academic purposes as part of the ML Capstone Project.

---

**Built with ❤️ by Team TASKMASTERS**
