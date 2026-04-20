import os
import re
import pickle
from contextlib import asynccontextmanager
 
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
 
# ── Globals ──────────────────────────────────────────────────────────────────
tfidf      = None
svm        = None
bert_model = None
tokenizer  = None
 
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
 
 
# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup, release on shutdown."""
    global tfidf, svm, bert_model, tokenizer
 
    # SVM + TF-IDF
    try:
        if os.path.exists("tfidf_vectorizer.pkl"):
            with open("tfidf_vectorizer.pkl", "rb") as f:
                tfidf = pickle.load(f)
        if os.path.exists("svm_model.pkl"):
            with open("svm_model.pkl", "rb") as f:
                svm = pickle.load(f)
        if tfidf and svm:
            print("✓ SVM + TF-IDF loaded")
        else:
            print("⚠ SVM model files not found — SVM endpoint will be unavailable")
    except Exception as e:
        print(f"⚠ Could not load SVM models: {e}")
 
    # DistilBERT
    try:
        tokenizer  = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        bert_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3
        )
        if os.path.exists("best_distilbert.pt"):
            bert_model.load_state_dict(
                torch.load("best_distilbert.pt", map_location="cpu")
            )
            print("✓ DistilBERT loaded with fine-tuned weights")
        else:
            print("⚠ best_distilbert.pt not found — using base weights")
        bert_model.eval()
    except Exception as e:
        print(f"⚠ Could not load BERT model: {e}")
        print("  The app will serve SVM predictions only")
 
    yield  # application runs here
 
    # Cleanup (optional but clean)
    bert_model = None
    tokenizer  = None
    print("Models released")
 
 
# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sentiment Classifier", lifespan=lifespan)
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s!?.,'\"-]", "", text)
    return re.sub(r"\s+", " ", text).strip()
 
 
# ── Schema ────────────────────────────────────────────────────────────────────
class ReviewRequest(BaseModel):
    text:  str
    model: str = "bert"   # "bert" | "svm"
 
 
# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "models_loaded": {
            "svm":  svm is not None and tfidf is not None,
            "bert": bert_model is not None,
        },
    }
 
 
@app.post("/predict")
def predict(req: ReviewRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Input text must not be empty.")
 
    cleaned = clean_text(req.text)
 
    # ── SVM branch ────────────────────────────────────────────────────────────
    if req.model == "svm":
        if svm is None or tfidf is None:
            raise HTTPException(
                status_code=503,
                detail="SVM model is not loaded. Check that tfidf_vectorizer.pkl and svm_model.pkl exist.",
            )
        vec  = tfidf.transform([cleaned])
        pred = svm.predict(vec)[0]
 
        # Normalise: SVM might return int or string labels
        if isinstance(pred, (int, float)):
            sentiment = LABEL_MAP.get(int(pred), str(pred))
        else:
            sentiment = str(pred)
 
        return {"model": "SVM", "sentiment": sentiment}
 
    # ── BERT branch ───────────────────────────────────────────────────────────
    if bert_model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="BERT model is not loaded. Check that best_distilbert.pt exists and torch/transformers are installed.",
        )
 
    tokens = tokenizer(
        cleaned,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length",
    )
 
    # Warn caller when input was truncated
    original_token_count = len(tokenizer.encode(cleaned, add_special_tokens=True))
    truncated = original_token_count > 128
 
    with torch.no_grad():
        logits = bert_model(**tokens).logits
 
    pred       = logits.argmax(dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()
 
    response = {
        "model":      "DistilBERT",
        "sentiment":  LABEL_MAP[pred],
        "confidence": round(confidence, 4),
    }
    if truncated:
        response["warning"] = (
            f"Input was {original_token_count} tokens and was truncated to 128."
        )
    return response
