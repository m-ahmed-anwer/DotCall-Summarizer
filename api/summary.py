from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from gradio_client import Client

candidate_labels = [
    "politics", "technology", "health", "sports", "entertainment", "finance", "education",
    "science", "environment", "business", "economy", "law", "history", "arts", "culture",
    "religion", "philosophy", "psychology", "sociology", "literature", "music", "film",
    "television", "theater", "food", "travel", "fashion", "beauty", "gaming", "automotive",
    "real estate", "investing", "marketing", "advertising", "media",
    "journalism", "publishing", "nonprofit", "government", "military", "space", "weather",
    "animals", "agriculture", "architecture", "construction", "design", "engineering",
    "aviation", "maritime", "transportation", "logistics", "manufacturing", "retail",
    "hospitality", "tourism", "events", "human resources", "legal", "public relations",
    "consulting", "security", "cybersecurity", "cryptocurrency", "blockchain", "robotics",
    "artificial intelligence", "machine learning", "data science", "cloud computing",
    "internet of things", "virtual reality", "augmented reality", "biotechnology",
    "nanotechnology", "energy", "sustainability", "healthcare", "medicine", "pharmaceuticals",
    "biomedical", "psychotherapy", "wellness", "fitness", "nutrition", "parenting", "education",
    "elearning", "teaching", "training", "research", "development", "innovation",
    "entrepreneurship", "startups", "venture capital", "fundraising", "nonprofits", "volunteering",
    "community", "social impact", "ethics", "morality", "demography", "anthropology", "archaeology",
    "languages", "linguistics", "translation", "interpretation", "communication", "media",
    "film studies", "television studies", "gender studies", "cultural studies"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

summary_model_ckpt = "philschmid/bart-large-cnn-samsum"
topic_model_ckpt = "facebook/bart-large-mnli"
client = Client("valurank/Headline_generator")

tokenizer_topic = AutoTokenizer.from_pretrained(topic_model_ckpt)
model_topic = AutoModelForSequenceClassification.from_pretrained(topic_model_ckpt).to(device)

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
    #inputs = tokenizer_topic(transcription, return_tensors="pt", truncation=True).to(device)
    #outputs = model_topic(**inputs)
    #scores = outputs.logits.softmax(dim=-1).detach().cpu().numpy()[0]
    
    # Get top 3 topics
    #top_indices = scores.argsort()[-3:][::-1]
    #top_topics = [candidate_labels[i] for i in top_indices]
    top_topics = "title comes here from top_topics_convert"
    
    return top_topics
