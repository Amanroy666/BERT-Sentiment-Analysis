"""FastAPI inference endpoint"""
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from ..model.bert_classifier import BERTSentimentClassifier

app = FastAPI()

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict

model = BERTSentimentClassifier()
model.load_state_dict(torch.load('model.pt'))
model.eval()

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    # Tokenize and predict
    sentiment_labels = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
    
    with torch.no_grad():
        # Model inference logic here
        probs = [0.1, 0.15, 0.2, 0.35, 0.2]  # Example
        
    max_idx = probs.index(max(probs))
    
    return SentimentResponse(
        sentiment=sentiment_labels[max_idx],
        confidence=max(probs),
        probabilities=dict(zip(sentiment_labels, probs))
    )
