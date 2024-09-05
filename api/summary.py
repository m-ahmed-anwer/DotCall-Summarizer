from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from gradio_client import Client
from api.labels import candidate_labels


# Load the summarization model and tokenizer
summary_model_dir = "/Users/ahmed/Desktop/flant5basesumm1/checkpoint-1558"
summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_dir)
summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_dir)


topic_model_ckpt = "facebook/bart-large-mnli"
client = Client("valurank/Headline_generator")


classifier = pipeline("zero-shot-classification", model=topic_model_ckpt, device=0)



def summary_convert(transcription):
    prompt = f"Summarize the following conversation.\n\n{transcription}\n\nSummary:"
    inputs = summary_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = summary_model.generate(
        inputs['input_ids'],
        max_length=150,
        min_length=30,
        num_beams=4,
        early_stopping=True
    )
    summary = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def title_convert(transcription):
    title = client.predict(
        transcription,
        api_name="/predict"
    )
    return title

def top_topics_convert(transcription):
    topic_result = classifier(transcription, candidate_labels=candidate_labels,)
    topic_names = topic_result['labels'][:3]

    return topic_names
