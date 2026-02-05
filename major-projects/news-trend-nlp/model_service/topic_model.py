import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from typing import List, Dict, Any

class TopicModel:
    def __init__(self, n_topics: int = 15, max_features: int = 2000):
        self.n_topics = n_topics
        self.max_features = max_features
        # English stop words + common news words that might be noise
        self.stop_words = 'english' 
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=self.stop_words,
            min_df=2,  # Ignore terms that appear in less than 2 docs
            max_df=0.95 # Ignore terms that appear in > 95% of docs
        )
        self.nmf_model = NMF(
            n_components=self.n_topics,
            random_state=42,
            init='nndsvd', # Better initialization for speed/sparsity
            max_iter=200
        )

    def fit_transform(self, documents: List[str], doc_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Clustering documents into topics on-the-fly.
        Returns a list of topics with their keywords and assigned document IDs.
        """
        if not documents:
            return []

        # 1. Vectorize
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        feature_names = self.vectorizer.get_feature_names_out()

        # 2. Fit NMF
        # W: Document-Topic Matrix (shape: [n_docs, n_topics])
        # H: Topic-Term Matrix (shape: [n_topics, n_features])
        W = self.nmf_model.fit_transform(tfidf_matrix)
        H = self.nmf_model.components_

        # 3. Extract Top Keywords per Topic
        topics = []
        for topic_idx, topic in enumerate(H):
            # Get top 10 words for this topic
            top_features_ind = topic.argsort()[:-11:-1]
            keywords = [feature_names[i] for i in top_features_ind]
            
            # Find documents that belong to this topic (dominant topic)
            # We look at W (n_docs x n_topics) and pick argmax for each doc
            assigned_docs = []
            
            topics.append({
                "topic_id": topic_idx,
                "keywords": keywords,
                "article_ids": [] # Will fill below
            })

        # 4. Assign Docs to Dominant Topic
        # W[i, j] is the association of doc i with topic j
        dominant_topics = np.argmax(W, axis=1)
        
        for i, topic_idx in enumerate(dominant_topics):
            # Optional: Threshold check. If association is too low, maybe classify as "Noise"?
            # For now, strictly assign to max.
            topics[topic_idx]["article_ids"].append(doc_ids[i])

        # Filter out empty topics? No, keep them but they will have 0 docs.
        # Actually better to filter empty ones to save DB space
        active_topics = [t for t in topics if t["article_ids"]]
        
        return active_topics
