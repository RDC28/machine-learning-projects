from rest_framework import serializers
from .models import TrendTopic, RunLog

class TrendTopicSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrendTopic
        fields = ['id', 'label', 'keywords', 'summary', 'article_count', 'representative_articles']

class RunLogSerializer(serializers.ModelSerializer):
    trends = TrendTopicSerializer(many=True, read_only=True)
    
    class Meta:
        model = RunLog
        fields = ['id', 'timestamp', 'status', 'articles_analyzed_count', 'trends']
