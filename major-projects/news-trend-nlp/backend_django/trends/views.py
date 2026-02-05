from rest_framework import viewsets, views
from rest_framework.response import Response
from .models import TrendTopic, RunLog
from .serializers import TrendTopicSerializer, RunLogSerializer
from .utils import run_trend_analysis
from django.shortcuts import render

# API Views
class TrendTopicViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint that allows trends to be viewed.
    """
    queryset = TrendTopic.objects.all().order_by('-run__timestamp')
    serializer_class = TrendTopicSerializer

class LatestTrendsView(views.APIView):
    """
    Returns trends from the most recent successful run.
    """
    def get(self, request):
        latest_run = RunLog.objects.filter(status="SUCCESS").order_by('-timestamp').first()
        if not latest_run:
            return Response({"message": "No data available yet."}, status=404)
        
        trends = latest_run.trends.all().order_by('-article_count')
        serializer = TrendTopicSerializer(trends, many=True)
        return Response({
            "run_id": latest_run.id,
            "timestamp": latest_run.timestamp,
            "trends": serializer.data
        })

class TriggerRunView(views.APIView):
    """
    Triggers a new analysis run manually.
    """
    def post(self, request):
        run = run_trend_analysis()
        return Response({
            "run_id": run.id,
            "status": run.status,
            "articles": run.articles_analyzed_count
        })

# Template Views (Public Website)
def home_view(request):
    return render(request, 'trends/home.html')

def trends_view(request):
    return render(request, 'trends/trends.html')

def about_view(request):
    return render(request, 'trends/about.html')
