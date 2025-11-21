"""BERT model for sentiment classification"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERTSentimentClassifier(nn.Module):
    def __init__(self, n_classes=5, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped = self.dropout(pooled_output)
        logits = self.classifier(dropped)
        return logits
