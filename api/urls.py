from django.urls import path
from .views import summarize_dialogue_api_via_text
from .views import summarize_dialogue_api_via_audio

urlpatterns = [
      path('summarize/text', summarize_dialogue_api_via_text, name='summarize_dialogue'),
      path('summarize/audio', summarize_dialogue_api_via_audio, name='summarize_dialogue'),
]
