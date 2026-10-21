# Deployment Guide for ML Capstone App

## 🚀 Deployment Options

### 1. **Streamlit Community Cloud** (Recommended - FREE)

**Pros:**
- ✅ Completely free
- ✅ Easy setup (3 clicks)
- ✅ Auto-updates from GitHub
- ✅ HTTPS included
- ✅ No server management

**Steps:**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `Sukanth19/ML-Capstone-Review-1`
5. Set main file: `app.py`
6. Click "Deploy"

**Done!** Your app will be live at `https://[your-app-name].streamlit.app`

---

### 2. **Render** (FREE tier available)

**Pros:**
- ✅ Free tier with 750 hours/month
- ✅ Auto-deploy from GitHub
- ✅ Custom domains
- ✅ Good performance

**Steps:**
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New+" → "Web Service"
4. Connect your repository
5. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
6. Click "Create Web Service"

---

### 3. **Railway** (FREE $5 credit monthly)

**Pros:**
- ✅ $5/month free credit
- ✅ Fast deployment
- ✅ GitHub integration
- ✅ Easy scaling

**Steps:**
1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repo
5. Add environment variables if needed
6. Railway will auto-detect and deploy

---

### 4. **Heroku** (Paid - $5-7/month)

**Pros:**
- ✅ Reliable and mature platform
- ✅ Easy scaling
- ✅ Good documentation

**Steps:**
1. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

2. Create `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

3. Deploy:
```bash
heroku login
heroku create your-app-name
git push heroku main
```

---

### 5. **AWS EC2** (Variable cost - ~$5-10/month for t2.micro)

**Pros:**
- ✅ Full control
- ✅ Scalable
- ✅ Free tier for 12 months

**Steps:**
1. Launch EC2 instance (Ubuntu)
2. SSH into instance
3. Install dependencies:
```bash
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
```
4. Run with PM2 or systemd:
```bash
streamlit run app.py --server.port=8501
```
5. Set up nginx reverse proxy (optional)

---

### 6. **Google Cloud Run** (Pay per use - very cheap)

**Pros:**
- ✅ Serverless (only pay when used)
- ✅ Auto-scaling
- ✅ Very affordable for low traffic

**Steps:**
1. Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Deploy:
```bash
gcloud run deploy ml-capstone --source .
```

---

## 📊 Comparison Table

| Platform | Cost | Ease | Performance | Best For |
|----------|------|------|-------------|----------|
| **Streamlit Cloud** | FREE | ⭐⭐⭐⭐⭐ | Good | Quick demos, portfolios |
| **Render** | FREE* | ⭐⭐⭐⭐ | Good | Small to medium apps |
| **Railway** | $5/mo | ⭐⭐⭐⭐ | Good | Modern deployment |
| **Heroku** | $5-7/mo | ⭐⭐⭐ | Good | Production apps |
| **AWS EC2** | ~$10/mo | ⭐⭐ | Excellent | Full control needed |
| **Google Cloud Run** | Pay-per-use | ⭐⭐⭐ | Excellent | Variable traffic |

*Free tier with limitations

---

## 🎯 Recommended Choice

For your ML Capstone project, I recommend **Streamlit Community Cloud** because:
1. ✅ It's completely FREE
2. ✅ Perfect for data science/ML apps
3. ✅ Takes literally 2 minutes to deploy
4. ✅ Automatically updates when you push to GitHub
5. ✅ Professional-looking URL
6. ✅ Great for showcasing in your portfolio

---

## 📝 Additional Files Needed

### For Streamlit Cloud
**No additional files needed!** Just push your code.

### For Render/Railway
Create `.streamlit/config.toml`:
```toml
[server]
headless = true
port = 8501
enableCORS = false

[browser]
gatherUsageStats = false
```

### For Docker-based (Cloud Run, etc.)
See Dockerfile above.

---

## 🔧 Pre-Deployment Checklist

- [x] Test app locally: `streamlit run app.py`
- [ ] Ensure all CSV files are in the repo
- [ ] Check requirements.txt has all dependencies
- [ ] Remove any API keys or secrets from code
- [ ] Test with sample data
- [ ] Update README with live demo link

---

## 🌐 Post-Deployment

After deploying, update your README.md:

```markdown
## 🔗 Live Demo
View the deployed application: [https://your-app.streamlit.app](https://your-app.streamlit.app)
```

---

## 💡 Tips for Better Performance

1. **Cache data loading:**
```python
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')
```

2. **Optimize file sizes:**
   - Keep CSV files under 50MB
   - Compress images if any

3. **Monitor resources:**
   - Check memory usage
   - Optimize model training

---

## 📞 Need Help?

- Streamlit Docs: https://docs.streamlit.io/
- Community Forum: https://discuss.streamlit.io/
- GitHub Issues: Create an issue in your repo

---

**Ready to deploy? Start with Streamlit Cloud - it's the easiest!** 🚀
