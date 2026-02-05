import requests
import json
import os
from datetime import datetime
from .models import RunLog, TrendTopic
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

# --- Headline Generation (HuggingFace Inference API) ---
def generate_headline(keywords, representative_texts=None):
    """
    Uses HuggingFace Inference API to generate a proper news headline.
    Falls back to formatted keywords if API unavailable.
    """
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        print("WARNING: No HF Token found. Using fallback labels.")
        return ", ".join(keywords[:3]).title()
    
    try:
        from huggingface_hub import InferenceClient
        
        # Use Qwen model - free, fast, and available on HF Inference API
        client = InferenceClient(
            model="Qwen/Qwen2.5-72B-Instruct",
            token=hf_token
        )
        
        # Create a prompt for headline generation
        topic_str = ", ".join(keywords[:5])
        prompt = f"Generate a short, professional news headline (max 10 words) about: {topic_str}. Just the headline, nothing else."
        
        # Generate using chat completions API
        result = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.7
        )
        
        # Extract the headline from response
        if result and result.choices:
            headline = result.choices[0].message.content.strip()
            # Remove any quotes that might be in the output
            headline = headline.strip('"\'')
            if len(headline) > 5:
                return headline
        
        return ", ".join(keywords[:3]).title()
        
    except Exception as e:
        # Silent fallback - don't spam console with errors
        # This handles rate limits, timeouts, and API issues gracefully
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
