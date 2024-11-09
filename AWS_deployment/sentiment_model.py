import boto3
import json
import os
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 functions
def download_data(s3_bucket, s3_key, local_file):
    s3 = boto3.client('s3')
    s3.download_file(s3_bucket, s3_key, local_file)
    logger.info("Data downloaded from S3.")

def upload_results(s3_bucket, s3_key, local_file):
    s3 = boto3.client('s3')
    s3.upload_file(local_file, s3_bucket, s3_key)
    logger.info("Data uploaded to S3.")

# Text preprocessing function
def preProcess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text)
    text = re.sub(r'@\w+', '[USER]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenization function for sentiment analysis
def tokenize_texts(texts, tokenizer, max_len=140):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

# Load models and tokenizers
model_dir = "model_dir"  # Adjust this path if using a custom model directory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load sentiment analysis model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_dir, use_safetensors=True).to(device)
model.eval()

# Load NER model and tokenizer for entity recognition
ner_tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
ner_model = AutoModelForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english').to(device)
ner_model.eval()

# Entity extraction function
def extract_entities(text):
    inputs = ner_tokenizer(text, return_tensors="pt").to(device)
    outputs = ner_model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)

    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    entities = []
    for token, pred in zip(tokens, predictions[0].cpu().numpy()):
        label = ner_model.config.id2label[pred]
        if label != 'O':  # Only consider tokens labeled as entities
            entities.append((token, label))

    return entities

# Main processing function
def main():
    # Retrieve environment variables for S3 bucket and keys
    data_bucket = os.getenv('DATA_BUCKET')
    data_key = os.getenv('DATA_KEY')
    results_bucket = os.getenv('RESULTS_BUCKET')
    results_key = os.getenv('RESULTS_KEY')

    # Define paths for temporary local storage
    local_data_file = '/tmp/reddit_data.json'  # Lambda's temporary storage directory
    local_results_file = '/tmp/sentiment_results.json'

    # Download data from S3
    download_data(data_bucket, data_key, local_data_file)

    # Load the data
    with open(local_data_file, 'r') as f:
        posts_data = json.load(f)

    # Analyze each post
    for post in posts_data:
        text = post.get('Title', '') + ' ' + post.get('Content', '')
        preprocessed_text = preProcess(text)

        # Tokenize and run sentiment analysis
        input_ids, attention_masks = tokenize_texts([preprocessed_text], tokenizer, max_len=140)
        input_ids, attention_masks = input_ids.to(device), attention_masks.to(device)

        label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_masks)
            sentiment = torch.argmax(outputs.logits, dim=1).item()
        post['Sentiment'] = label_mapping[sentiment]

        # Extract entities
        entities = extract_entities(text)
        post['Entities'] = entities  # Add entities as a list of (token, entity label) pairs

    # Save the results to a JSON file
    with open(local_results_file, 'w') as f:
        json.dump(posts_data, f, indent=4)

    logger.info("Sentiment and entities processed for each post and saved.")

    # Upload results to S3
    upload_results(results_bucket, results_key, local_results_file)

if __name__ == '__main__':
    main()
