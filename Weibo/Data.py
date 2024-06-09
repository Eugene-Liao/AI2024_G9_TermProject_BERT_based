import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm 
from tabulate import tabulate
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

def chunk_text(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    token_chunks = [tokens[i:i + max_length - 2] for i in range(0, len(tokens), max_length - 2)]  # Reserve space for special tokens
    return token_chunks

def tokenize_and_chunk_posts(posts, tokenizer, max_length=512):
    tokenized_chunks = []
    for post in posts:
        chunks = chunk_text(post, tokenizer, max_length)
        for chunk in chunks:
            encoded_chunk = tokenizer.encode_plus(
                chunk,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            tokenized_chunks.append(encoded_chunk)
    return tokenized_chunks

class SocialMediaDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        user_index = item['user_index']
        tokenized_chunks = tokenize_and_chunk_posts([item['post']], self.tokenizer, self.max_length)
        labels = torch.tensor(int(item['label']), dtype=torch.long)
        return user_index, tokenized_chunks, labels

class UserBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.user_indices = self.group_by_user(dataset)
        self.batches = self.create_batches()

    def group_by_user(self, dataset):
        user_indices = {}
        for idx in range(len(dataset)):
            user_index = dataset.data.iloc[idx]['user_index']
            if user_index not in user_indices:
                user_indices[user_index] = []
            user_indices[user_index].append(idx)
        return user_indices

    def create_batches(self):
        all_batches = []
        for user, indices in self.user_indices.items():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size:
                    # Handle the remaining posts for a user
                    batch += [-1] * (self.batch_size - len(batch))  # Fill with dummy indices
                all_batches.append(batch)
        np.random.shuffle(all_batches)
        return all_batches

    def __len__(self):
        return len(self.batches) 

    def __iter__(self):
        for batch in self.batches:
            yield batch
            
def custom_collate(tokenizer):
    def collate_fn(batch):
        # Filter out empty or dummy indices
        batch = [item for item in batch if item[1]]  # Ensure tokenized_chunks is not empty
    
        if not batch:
            # Handle case where the entire batch might be empty after filtering
            return [], {'input_ids': torch.tensor([]), 'attention_mask': torch.tensor([]), 'token_type_ids': torch.tensor([])}, torch.tensor([])
    
        batch_user_indices = [item[0] for item in batch]
        batch_tokenized_chunks = [item[1] for item in batch]
        batch_labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    
        input_ids = [chunk['input_ids'].squeeze(0) for user_chunks in batch_tokenized_chunks for chunk in user_chunks]
        attention_masks = [chunk['attention_mask'].squeeze(0) for user_chunks in batch_tokenized_chunks for chunk in user_chunks]
        if 'token_type_ids' in batch_tokenized_chunks[0][0]:
            token_type_ids = [chunk['token_type_ids'].squeeze(0) for user_chunks in batch_tokenized_chunks for chunk in user_chunks]
        else:
            token_type_ids = None
    
        # Pad sequences to the same length
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        if token_type_ids is not None:
            token_type_ids_padded = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        else:
            token_type_ids_padded = None
    
        # Prepare the batch data
        batch_data = {
            'input_ids': input_ids_padded,
            'attention_mask': attention_masks_padded,
            'token_type_ids': token_type_ids_padded if token_type_ids is not None else None
        }
    
        return batch_user_indices, batch_data, batch_labels
    return collate_fn