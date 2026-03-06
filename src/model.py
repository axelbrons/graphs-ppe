# src/model.py
import torch.nn as nn
from transformers import AutoModel

class GraphCodeBERTClassifier(nn.Module):
    def __init__(self, model_name, num_labels, freeze_backbone=False):
        super(GraphCodeBERTClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        # attention_mask here is the 4D mask (batch, 1, seq_len, seq_len)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
