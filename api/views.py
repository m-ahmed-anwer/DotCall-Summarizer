# summary/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import json

# Check if CUDA is available and use GPU if it is
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model checkpoint
model_ckpt = "google/pegasus-cnn_dailymail"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

# Create summarization pipeline
summarization_pipeline = pipeline("summarization", model=model_pegasus, tokenizer=tokenizer)

@csrf_exempt
def summarize_dialogue_api(request):
    if request.method == "POST":
        try:
            # Parse JSON data from request body
            data = json.loads(request.body)
            dialogue = data.get("dialogue", "")
            if not dialogue:
                return JsonResponse({"error": "Dialogue not provided"}, status=400)
            
            # Generate summary
            summary = summarization_pipeline(dialogue, max_length=300, num_beams=8, length_penalty=0.8)[0]['summary_text']
            
            # Replace '<n>' with actual new lines '\n'
            summary = summary.replace('<n>', '\n')
            
            return JsonResponse({"summary": summary}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)
