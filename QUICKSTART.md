# 🚀 Quick Start Guide

## Get Your ML Capstone App Running in 3 Steps!

### Step 1: Install Dependencies (30 seconds)
Open Command Prompt in this directory and run:
```bash
pip install -r requirements.txt
```

### Step 2: Launch the Application (5 seconds)
```bash
streamlit run app.py
```

OR double-click the `run_app.bat` file!

### Step 3: Use the Application
The app will open automatically in your browser at `http://localhost:8501`

---

## ⚡ What You Get

### 🎾 Classification Module
- **5 Algorithms**: Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM
- **Dataset**: Australian Open Tennis (1905 matches)
- **Prediction**: Will the higher-ranked player win?
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Visualizations**: Confusion matrices, ROC curves

### ⛳ Regression Module
- **10 Algorithms**: Linear, Ridge, Lasso, ElasticNet, Polynomial, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN
- **Dataset**: PGA Golf 2022 (1387 observations)
- **Prediction**: Golfer's average score
- **Metrics**: R², RMSE, MAE, MSE
- **Visualizations**: Predicted vs Actual, Residual plots

### 🔍 Clustering Module
- **Algorithm**: K-Means
- **Analysis**: Elbow method, Silhouette scores, PCA visualization

---

## 🎯 Quick Demo

1. Click "🎾 Classification (Tennis)" in sidebar
2. Click "🚀 Train All Classification Models"
3. Watch the magic happen! (10 seconds)
4. See all 5 models trained with full metrics and visualizations

---

## 📁 Required Files

Make sure these are in your directory:
- ✅ `app.py` (main application)
- ✅ `AustralianOpen.csv` (tennis data)
- ✅ `PGA2022.csv` (golf data)
- ✅ `requirements.txt` (dependencies)

---

## 💡 Pro Tips

- **First Time?** Training takes 10-15 seconds for all models
- **Best Browser**: Chrome or Firefox
- **Screen Size**: Desktop/laptop recommended
- **Stop App**: Press `Ctrl+C` in terminal

---

## 🆘 Need Help?

**Error: "Could not load data"**
→ Check that CSV files are in the same folder as app.py

**Error: "Module not found"**
→ Run `pip install -r requirements.txt` again

**App won't start**
→ Check if port 8501 is free, or use: `streamlit run app.py --server.port 8502`

---

## 🎓 Features Highlight

✨ **Interactive UI** - Click, explore, analyze  
📊 **Real-time Training** - Train on demand  
📈 **Rich Visualizations** - Professional plots  
🔍 **Model Comparison** - See which performs best  
🏆 **Best Model Identification** - Automatic selection  
💾 **Reproducible** - Fixed random seed (42)  

---

**Ready? Run this command:**
```bash
streamlit run app.py
```

**That's it! Enjoy your ML Capstone Application! 🎉**
