from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from gradio_client import Client
from api.labels import candidate_labels


summary_model_ckpt = "philschmid/bart-large-cnn-samsum"
topic_model_ckpt = "facebook/bart-large-mnli"
client = Client("valurank/Headline_generator")


classifier = pipeline("zero-shot-classification", model=topic_model_ckpt, device=0)

#tokenizer_topic = AutoTokenizer.from_pretrained(topic_model_ckpt)
#model_topic = AutoModelForSequenceClassification.from_pretrained(topic_model_ckpt)


def summary_convert(transcription):
    summarizer = pipeline("summarization", model=summary_model_ckpt)
    summary = summarizer(transcription)[0]['summary_text']
    return summary

def title_convert(transcription):
    title = client.predict(
        transcription,
        api_name="/predict"
    )
    return title

def top_topics_convert(transcription):
    result = classifier(transcription, candidate_labels, multi_label=True)

    # Extract the top 3 topics
    top_topics = sorted(zip(result["labels"], result["scores"]), key=lambda x: x[1], reverse=True)[:3]

    return top_topics
