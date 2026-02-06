# 🚀 Deployment Guide: News Trend NLP

This guide outlines how to deploy your **News Trend NLP** project.
All services (Backend, Model Service, and Frontend) will be hosted in the cloud.

---

## ✅ Step 1: Prepare GitHub Repository
(Already done if you are reading this on GitHub)

```bash
git init
git add .
git commit -m "Initial commit"
# Connect to your repo
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/news-trend-nlp.git
git push -u origin main
```

---

## 🚀 Step 2: Deploy Model Service (on Render)
This service runs the Topic Modeling (NMF).

1.  Go to [Render.com](https://render.com) and click **New +** -> **Web Service**.
2.  Connect your `news-trend-nlp` repository.
3.  **Configuration**:
    *   **Name**: `news-trend-model`
    *   **Root Directory**: `model_service`
    *   **Environment**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 10000`
    *   **Instance Type**: Free
4.  **Click Create Web Service**.
5.  **Copy the URL** (e.g., `https://news-trend-model.onrender.com`).

---

## 🛠️ Step 3: Deploy Backend API (on Render)
This runs the Django logic and database.

1.  Go to [Render.com](https://render.com) and click **New +** -> **Web Service**.
2.  Connect your GitHub repo.
3.  **Configuration**:
    *   **Name**: `news-trend-backend`
    *   **Root Directory**: `backend_django`
    *   **Environment**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate`
    *   **Start Command**: `gunicorn config.wsgi:application`
    *   **Instance Type**: Free
4.  **Environment Variables**:
    *   `PYTHON_VERSION`: `3.11.0`
    *   `SECRET_KEY`: (Generate a random string)
    *   `DEBUG`: `False`
    *   `MODEL_SERVICE_URL`: Paste the **Model Service URL** from Step 2 (e.g. `https://news-trend-model.onrender.com`).
    *   `HUGGINGFACEHUB_API_TOKEN`: (Your HF Token)
5.  **Click Create Web Service**.
6.  **Copy the URL** (e.g., `https://news-trend-backend.onrender.com`).

---

## 🌐 Step 4: Deploy Frontend (on Streamlit Cloud)
Streamlit apps work best on their native cloud.

1.  Go to [share.streamlit.io](https://share.streamlit.io).
2.  Click **New App**.
3.  Select Repository `news-trend-nlp`.
4.  **Main file path**: `frontend_streamlit/app.py`.
5.  **🚨 CRITICAL STEP: Configure Secrets**
    Go to **App Settings** -> **Secrets** and add:

    ```toml
    DJANGO_API_URL = "https://news-trend-backend.onrender.com/api/latest/"
    TRIGGER_RUN_URL = "https://news-trend-backend.onrender.com/api/run/"
    DJANGO_BASE_URL = "https://news-trend-backend.onrender.com"
    MODEL_SERVICE_URL = "https://news-trend-model.onrender.com"
    ```
    *(Replace with your actual URLs if they differ)*.

6.  **Click Save / Deploy**.

---

## 🎉 Done!
Your app is fully distributed:
*   **Calculations**: Render (Web Service)
*   **Logic/API**: Render (Web Service)
*   **UI**: Streamlit Cloud

### Troubleshooting
*   **"Conn Error" or Red Pills**: This means Streamlit cannot reach your services. Ensure the **Secrets** in Step 4 are 100% correct and the services on Render are not sleeping (the app will try to wake them up automatically).

