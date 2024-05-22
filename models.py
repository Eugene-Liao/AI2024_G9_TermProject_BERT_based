import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForSequenceClassification


class BertForUserClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForUserClassification, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits