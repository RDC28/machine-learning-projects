from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from topic_model import TopicModel

app = FastAPI(title="News Trend Model Service")

class ArticleInput(BaseModel):
    id: str
    text: str

class TopicResponse(BaseModel):
    topic_id: int
    keywords: List[str]
    article_ids: List[str]

class PredictRequest(BaseModel):
    articles: List[ArticleInput]
    num_topics: int = 15

@app.get("/")
def health_check():
    return {"status": "ok", "service": "model-service"}

@app.post("/predict_topics")
def predict_topics(request: PredictRequest):
    articles = request.articles
    num_topics = request.num_topics

    if not articles:
        return {"topics": []}
    
    # Extract text and IDs
    doc_texts = [a.text for a in articles]
    doc_ids = [a.id for a in articles]

    # Run Topic Modeling
    # Note: For a production system with varying load, we might instantiate this globally 
    # if we were loading a pre-trained model. Since we do "dynamic topic modeling" 
    # (clustering the *current* batch), we fit fresh every time.
    model = TopicModel(n_topics=num_topics)
    topics = model.fit_transform(documents=doc_texts, doc_ids=doc_ids)

    return {"topics": topics}
