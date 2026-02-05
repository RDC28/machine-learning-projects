import requests
import json
import os
from datetime import datetime
from .models import RunLog, TrendTopic
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from django.conf import settings

# --- GDELT Fetcher ---
def fetch_gdelt_data(max_records=100):
    """
    Fetches latest English news articles from GDELT Doc 2.0 API.
    """
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": "sourcelang:english",
        "mode": "artlist",
        "maxrecords": max_records,
        "format": "json",
        "sort": "DateDesc"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # GDELT returns {"articles": [...]}
        articles = data.get("articles", [])
        
        # Simplified list for our Model Service
        # We need unique IDs. GDELT usually has URL as unique enough or we use index.
        cleaned_articles = []
        for i, art in enumerate(articles):
            title = art.get("title", "")
            # Filter out very short titles or weird ones
            if len(title) > 20: 
                cleaned_articles.append({
                    "id": str(i), 
                    "text": title, # We use Title for clustering to save RAM. Full text scraping is too heavy & slow.
                    "url": art.get("url"),
                    "source": art.get("domain")
                })
        
        return cleaned_articles
        
    except Exception as e:
        print(f"Error fetching GDELT: {e}")
        return []

# --- Model Service Interaction ---
def get_topics_from_service(articles):
    """
    Sends articles to FastAPI Model Service for clustering.
    """
    url = f"{os.getenv('MODEL_SERVICE_URL', 'http://localhost:8001')}/predict_topics"
    payload = {
        "articles": [{"id": a["id"], "text": a["text"]} for a in articles],
        "num_topics": 10 # Configurable
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("topics", [])
    except Exception as e:
        print(f"Error calling Model Service: {e}")
        return []

# --- Summarization (LangChain) ---
def generate_headline(keywords):
    """
    Uses HF Inference API to generate a short news headline.
    """
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        print("WARNING: No HF Token found. Using fallback labels.")
        return ", ".join(keywords[:3]).title()
    
    # Try using a widely widely available model, but gracefully fail to keywords if unavailable
    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    prompt = f"Make a short news headline about: {', '.join(keywords)}"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 25, 
            "temperature": 0.6,
            "do_sample": True
        }
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=5)
        
        # If API is standard 410 Gone or 404, just return keywords silently
        if response.status_code in [404, 410, 503]:
            # print(f"Model unavailable ({response.status_code}), falling back to keywords.")
            return ", ".join(keywords[:3]).title()

        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            headline = result[0].get('generated_text', '')
            return headline.strip()
        
        return ", ".join(keywords[:3]).title()
        
    except Exception:
        # Silently fall back if anything goes wrong (timeouts, network, etc.)
        # This prevents the red error text spam in the console
        return ", ".join(keywords[:3]).title()

# --- Orchestrator ---
def run_trend_analysis():
    """
    Main function to run the full pipeline and save to DB.
    """
    run = RunLog.objects.create(status="RUNNING")
    
    try:
        # 1. Fetch
        articles = fetch_gdelt_data(max_records=150)
        run.articles_analyzed_count = len(articles)
        run.save()
        
        if not articles:
            run.status = "FAILED"
            run.save()
            return run
            
        # 2. Cluster
        topics_data = get_topics_from_service(articles)
        
        # 3. Process & Save
        for t in topics_data:
            t_keywords = t.get("keywords", [])
            t_article_ids = t.get("article_ids", [])
            
            if not t_article_ids:
                continue
            
            # Generate Headline using LLM
            label = generate_headline(t_keywords)
            
            # Use same fallback/logic for summary if needed, or simple string
            summary = f"News involving {', '.join(t_keywords[:5])}."
            
            # Find representative examples
            # Map back IDs to article objects
            examples = [
                a for a in articles 
                if a["id"] in t_article_ids
            ][:3] # Keep top 3
            
            TrendTopic.objects.create(
                run=run,
                label=label,
                keywords=json.dumps(t_keywords),
                summary=summary,
                article_count=len(t_article_ids),
                representative_articles=json.dumps(examples)
            )
            
        run.status = "SUCCESS"
        run.save()
        return run

    except Exception as e:
        run.status = "FAILED"
        run.save()
        print(f"Pipeline failed: {e}")
        return run
