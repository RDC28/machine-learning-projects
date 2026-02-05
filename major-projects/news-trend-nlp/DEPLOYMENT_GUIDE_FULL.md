# 🚀 Deployment Guide: News Trend NLP

This guide outlines how to deploy your **News Trend NLP** project using a hybrid approach to maximize free tier benefits:
1.  **Model Service (FastAPI)** -> **Vercel** (Serverless, fast, free).
2.  **Backend API (Django)** -> **Render** (Free Web Service).
3.  **Frontend (Streamlit)** -> **Streamlit Cloud** (Best for Streamlit) OR **Render** (Alternative).

---

## ✅ Step 1: Prepare GitHub Repository
Since you haven't set up the repo, follow these steps in your terminal (inside the project folder):

```bash
# 1. Initialize Git
git init

# 2. Add all files
git add .

# 3. Create initial commit
git commit -m "Initial commit for deployment"

# 4. Create a repository on GitHub.com (e.g., 'news-trend-nlp')
# 5. Connect and push (replace YOUR_USERNAME)
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/news-trend-nlp.git
git push -u origin main
```

---

## 🚀 Step 2: Deploy Model Service (on Vercel)
This service runs the Topic Modeling (NMF).

1.  Go to [Vercel.com](https://vercel.com) and **Add New Project**.
2.  Import your `news-trend-nlp` repository.
3.  **Critical Configuration**:
    *   **Framework Preset**: Select "Other" (or let it detect).
    *   **Root Directory**: Click "Edit" and select `model_service`.
4.  **Click Deploy**.
5.  Wait for deployment. specific URL will be generated (e.g., `https://news-trend-model-xyz.vercel.app`).
6.  **Copy this URL**. You will need it for the Backend.

---

## 🛠️ Step 3: Deploy Backend API (on Render)
This runs the Django logic and database.

1.  Go to [Render.com](https://render.com) and click **New +** -> **Web Service**.
2.  Connect your GitHub repo.
3.  **Configuration**:
    *   **Name**: `news-trend-backend`
    *   **Root Directory**: `backend_django` (Important!)
    *   **Environment**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt && python manage.py collectstatic --noinput`
        *   *(Note: Migration command `python manage.py migrate` is also good to add, e.g., create a `build.sh` script, but for now manual migration or including it in build command works)*.
    *   **Start Command**: `gunicorn config.wsgi:application`
    *   **Instance Type**: Free
4.  **Environment Variables** (Scroll down to "Advanced"):
    *   `PYTHON_VERSION`: `3.11.0`
    *   `SECRET_KEY`: (Generate a random string)
    *   `DEBUG`: `False`
    *   `MODEL_SERVICE_URL`: Paste the **Vercel URL** from Step 2 (no trailing slash, e.g. `https://...vercel.app`).
    *   `HUGGINGFACEHUB_API_TOKEN`: (Your HF Token from .env)
5.  **Click Create Web Service**.
6.  Wait for it to go live. Content might be empty initially.
7.  **Copy the Render Service URL** (e.g., `https://news-trend-backend.onrender.com`).

*Note on Database*: By default, this uses SQLite which resets every time the app restarts (approx 15 mins of inactivity). For a permanent DB, add a **PostgreSQL** instance on Render and copy the `INTERNAL_DATABASE_URL` to a `DATABASE_URL` env var in this service.

---

## 🌐 Step 4: Deploy Frontend (on Streamlit Cloud)
Streamlit apps work best on their native cloud.

1.  Go to [share.streamlit.io](https://share.streamlit.io).
2.  Click **New App**.
3.  Select Repository `news-trend-nlp`.
4.  **Main file path**: `frontend_streamlit/app.py`.
5.  **Advanced Settings -> Secrets** (Env vars):
    Add the following TOML:
    ```toml
    DJANGO_API_URL = "https://YOUR-RENDER-BACKEND-URL.onrender.com/api/latest/"
    TRIGGER_RUN_URL = "https://YOUR-RENDER-BACKEND-URL.onrender.com/api/run/"
    ```
    *(Replace with your actual Render URL from Step 3)*.
6.  **Click Deploy**.

---

## 🎉 Done!
Your app is fully distributed:
*   **Calculations**: Vercel (Serverless)
*   **Logic/API**: Render (Web Service)
*   **UI**: Streamlit Cloud

### Troubleshooting
*   **500 Errors**: Check the Render logs.
*   **Model Service Error**: Check Vercel logs/Function logs.
*   **CORS**: We enabled `CORS_ALLOW_ALL_ORIGINS = True` in settings, so cross-domain requests should work fine.
