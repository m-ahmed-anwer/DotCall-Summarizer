from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2FeatureExtractor
import torch

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

# Check if CUDA is available and use GPU if it is
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model checkpoints
summary_model_ckpt = "google/pegasus-cnn_dailymail"
topic_model_ckpt = "facebook/bart-large-mnli"

# Load tokenizer and model for summarization
tokenizer_summary = AutoTokenizer.from_pretrained(summary_model_ckpt)
model_summary = AutoModelForSeq2SeqLM.from_pretrained(summary_model_ckpt).to(device)

# Load tokenizer and model for topic classification
tokenizer_topic = AutoTokenizer.from_pretrained(topic_model_ckpt)
model_topic = AutoModelForSequenceClassification.from_pretrained(topic_model_ckpt).to(device)

# Create summarization pipeline
summarization_pipeline = pipeline("summarization", model=model_summary, tokenizer=tokenizer_summary, device=0 if device == "cuda" else -1)

# Create topic classification pipeline
topic_classification_pipeline = pipeline("zero-shot-classification", model=model_topic, tokenizer=tokenizer_topic, device=0 if device == "cuda" else -1)



def generate_summarize(transcription):

    # Generate summary
    summary = summarization_pipeline(transcription, max_length=248, num_beams=8, length_penalty=0.8)[0]['summary_text']
    summary = summary.replace('<n>', '\n')

    # Generate topic
    topic_result = topic_classification_pipeline(transcription, candidate_labels=candidate_labels)
    top_topics = topic_result['labels'][:3]
    
    return summary, top_topics

