@echo off
echo Starting News Trend NLP System...

start "Model Service (FastAPI)" cmd /k "cd model_service && ..\.venv\Scripts\uvicorn.exe main:app --reload --port 8001"
timeout /t 3

start "Backend (Django)" cmd /k "cd backend_django && ..\.venv\Scripts\python.exe manage.py runserver 8000"
timeout /t 5

start "Frontend (Streamlit)" cmd /k "cd frontend_streamlit && ..\.venv\Scripts\streamlit.exe run app.py"

echo All services started!
echo Django: http://localhost:8000
echo Streamlit: http://localhost:8501
pause
