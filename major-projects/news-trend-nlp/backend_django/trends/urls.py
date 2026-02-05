from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TrendTopicViewSet, LatestTrendsView, TriggerRunView, home_view, trends_view, about_view

router = DefaultRouter()
router.register(r'all', TrendTopicViewSet)

urlpatterns = [
    # Pages
    path('', home_view, name='home'),
    path('explore/', trends_view, name='trends'),
    path('about/', about_view, name='about'),
    
    # API
    path('api/', include(router.urls)),
    path('api/latest/', LatestTrendsView.as_view(), name='latest_trends'),
    path('api/run/', TriggerRunView.as_view(), name='trigger_run'),
]
