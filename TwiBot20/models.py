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

    def forward(self, input_ids, attention_mask, token_type_ids=None, return_embeddings=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1] #capture the [CLS] token
        if return_embeddings:
            return pooled_output
                
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class CombinedModel(nn.Module):
    def __init__(self, bert_embedding_dim, metadata_dim, hidden_dim, output_dim):
        super(CombinedModel, self).__init__()
        self.bert_embedding_dim = bert_embedding_dim
        self.metadata_dim = metadata_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(bert_embedding_dim + metadata_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, bert_embeddings, metadata_features):
        combined = torch.cat((bert_embeddings, metadata_features), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        logits = self.classifier(x)
        
        
        return logits