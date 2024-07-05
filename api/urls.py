from django.urls import path
from .views import summarize_dialogue_api

urlpatterns = [
      path('summarize/', summarize_dialogue_api, name='summarize_dialogue'),
]
