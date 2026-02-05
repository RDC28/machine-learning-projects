# News Trend NLP

## Project Overview

News Trend NLP is a full-stack automated intelligence system designed to scrape, cluster, and analyze global news narratives in real-time. Unlike traditional news aggregators that simply list articles, this system uses Natural Language Processing (NLP) to identify underlying themes and group related stories into coherent "trends."

The system operates by continuously fetching data from the GDELT Project (Global Database of Events, Language, and Tone), processing article titles using Non-Negative Matrix Factorization (NMF) for topic modeling, and generating concise headlines using Large Language Models (LLM).

## System Architecture

The project follows a microservices architecture to separate concerns between data processing, machine learning, and user interface.

1.  **Backend API (Django REST Framework)**: The central orchestrator. It manages the database, triggers analysis runs, fetches raw data from GDELT, and serves the results via a REST API.
2.  **Model Service (FastAPI)**: A dedicated, lightweight serverless function responsible solely for the heavy lifting of text vectorization (TF-IDF) and clustering (NMF). It is stateless and optimized for speed.
3.  **Frontend (Streamlit)**: A distinct dashboard interface that consumes the Backend API to visualize trends, metrics, and source articles for the end user.
4.  **External Services**:
    *   **GDELT Project**: Source of raw news metadata.
    *   **Hugging Face Inference API**: Used for zero-shot summarization and headline generation.

## Technical Stack

*   **Language**: Python 3.9+
*   **Backend Framework**: Django 5.0, Django REST Framework
*   **ML Service Framework**: FastAPI, Uvicorn
*   **Frontend Framework**: Streamlit
*   **Machine Learning**: Scikit-learn (TF-IDF, NMF), Numpy
*   **Database**: SQLite (Development), PostgreSQL (Production)
*   **Deployment**:
    *   Backend: Render (Web Service)
    *   Model Service: Vercel (Serverless)
    *   Frontend: Streamlit Cloud

## Folder Structure

*   `backend_django/`: Contains the Django project.
    *   `trends/`: Main application logic (fetching, analyzing, serving APIs).
    *   `config/`: Project-level settings and routing.
*   `model_service/`: Contains the FastAPI project.
    *   `main.py`: Entry point for the ML inference API.
    *   `topic_model.py`: Logic for Scikit-learn clustering pipelines.
*   `frontend_streamlit/`: Contains the dashboard application.
    *   `app.py`: Main Streamlit interface.
    *   `style.css`: Custom styling for the dashboard.
*   `DEPLOYMENT_GUIDE_FULL.md`: Detailed step-by-step guide for deploying the system.

## Installation and Local Setup

### Prerequisites
*   Python 3.9 or higher
*   Git

### 1. Clone the Repository
```bash
git clone <repository_url>
cd news-trend-nlp
```

### 2. Setup Model Service
The backend requires the model service to be running to perform analysis.

```bash
cd model_service
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```
*The service will start at http://localhost:8001*

### 3. Setup Backend API
Open a new terminal window.

```bash
cd backend_django
pip install -r requirements.txt
```

Create a `.env` file in `backend_django/` with the following keys:
```env
SECRET_KEY=your_django_secret_key
DEBUG=True
MODEL_SERVICE_URL=http://localhost:8001
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

Run migrations and start the server:
```bash
python manage.py migrate
python manage.py runserver 8000
```
*The API will start at http://localhost:8000*

### 4. Setup Frontend Dashboard
Open a third terminal window.

```bash
cd frontend_streamlit
pip install -r requirements.txt
```

Create a `.env` file or set environment variables for the frontend:
```env
DJANGO_API_URL=http://localhost:8000/api/latest/
TRIGGER_RUN_URL=http://localhost:8000/api/run/
```

Launch the app:
```bash
streamlit run app.py
```

## API Endpoints

### Backend (Django)

*   `GET /api/latest/`: Returns the results of the most recent successful trend analysis, including topics, article counts, and timestamps.
*   `POST /api/run/`: Triggers a new analysis pipeline. This will fetch data from GDELT, send it to the Model Service, and save the results to the database.

### Model Service (FastAPI)

*   `POST /predict_topics`: Accepts a list of article titles and IDs. Returns cluster assignments and keywords for each detected topic.

## Usage

1.  Open the Streamlit dashboard in your browser.
2.  Navigate to the "Trend Explorer" page.
3.  Click "Refresh Intelligence" to trigger a new analysis cycle.
4.  View the generated trends, volume metrics, and representative headlines.
