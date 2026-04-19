# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle, re
import os

app = FastAPI(title="Sentiment Classifier")

# Load models at startup
tfidf = None
svm = None
bert_model = None
tokenizer = None

try:
    if os.path.exists('tfidf_vectorizer.pkl'):
        tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    if os.path.exists('svm_model.pkl'):
        svm = pickle.load(open('svm_model.pkl', 'rb'))
except Exception as e:
    print(f"Warning: Could not load pickle models: {e}")

try:
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    bert_model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=3)
    if os.path.exists('best_distilbert.pt'):
        bert_model.load_state_dict(torch.load('best_distilbert.pt', map_location='cpu'))
    bert_model.eval()
except Exception as e:
    print(f"Warning: Could not load BERT model: {e}")
    print("The app will work with SVM model only")

label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s!?.,\'"-]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

class ReviewRequest(BaseModel):
    text:  str
    model: str = 'bert'   # 'bert' or 'svm'

@app.post('/predict')
def predict(req: ReviewRequest):
    cleaned = clean_text(req.text)

    if req.model == 'svm':
        if svm is None or tfidf is None:
            return {'error': 'SVM model not loaded', 'status': 'model not available'}
        vec   = tfidf.transform([cleaned])
        pred  = svm.predict(vec)[0]
        return {'model': 'SVM', 'sentiment': pred}

    if bert_model is None or tokenizer is None:
        return {'error': 'BERT model not loaded', 'status': 'model not available'}
    
    import torch
    tokens = tokenizer(cleaned, return_tensors='pt',
                       max_length=128, truncation=True, padding='max_length')
    with torch.no_grad():
        logits = bert_model(**tokens).logits
    pred = logits.argmax(dim=1).item()
    conf = torch.softmax(logits, dim=1).max().item()
    return {'model': 'DistilBERT', 'sentiment': label_map[pred], 'confidence': round(conf, 4)}

@app.get('/')
def root():
    return {
        'status': 'running',
        'models_loaded': {
            'svm': svm is not None,
            'bert': bert_model is not None
        }
    }