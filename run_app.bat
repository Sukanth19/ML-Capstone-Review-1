@echo off
echo ================================
echo ML Capstone Project Application
echo Team TASKMASTERS (Team 16)
echo ================================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting application...
echo.
echo The application will open in your browser automatically.
echo If not, navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.
streamlit run app.py
