from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2FeatureExtractor
import torch
import json
import librosa
import os
from pydub import AudioSegment
from .summary import summary_convert,top_topics_convert,title_convert


# Load ASR model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


# Check if CUDA is available and use GPU if it is
device = "cuda" if torch.cuda.is_available() else "cpu"


@csrf_exempt
def summarize_dialogue_api_via_text(request):
    if request.method == "POST":
        try:
            # Parse JSON data from request body
            data = json.loads(request.body)
            dialogue = data.get("transcription", "")
            if not dialogue:
                return JsonResponse({"error": "Dialogue not provided"}, status=400)
            
            summary = summary_convert(dialogue)
            top_topics = top_topics_convert(dialogue)
            title= title_convert(dialogue)

            return JsonResponse({"transcription":dialogue, "summary": summary, "title":title, "topics":top_topics}, status=200)
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)

@csrf_exempt
def summarize_dialogue_api_via_audio(request):
    if request.method == 'POST' and request.FILES.get('file'):
        # Save the uploaded file
        audio_file = request.FILES['file']
        file_name = default_storage.save(audio_file.name, ContentFile(audio_file.read()))
        file_path = default_storage.path(file_name)

        # Load and process audio
        speech, rate = librosa.load(file_path, sr=16000)
        input_values = tokenizer(speech, return_tensors="pt").input_values

        # Perform transcription
        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.decode(predicted_ids[0])

        # Clean up: remove the saved file
        os.remove(file_path)
        
        summary = summary_convert(transcription)
        top_topics = top_topics_convert(transcription)
        title= title_convert(transcription)
     
        return JsonResponse({"transcription":transcription, "summary": summary,"title":title, "topics":top_topics}, status=200)
        
    else:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    

