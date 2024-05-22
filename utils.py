import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
import torch.nn as nn

from models import  BertForUserClassification
from tqdm import tqdm
from tabulate import tabulate
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

def dataProcess_simple(file_path, user_column, post_column, label):
    """
    Processes the raw data from a CSV file.
    
    Args:
    - file_path (str): Path to the CSV file.
    - user_column (str): Name of the column containing user indices.
    - post_column (str): Name of the column containing posts.
    - label_column (str): Name of the column to be added for labels.
    - label_value (int): Label value to assign to the processed data.
    
    Returns:
    - pd.DataFrame: Processed DataFrame containing user_index, posts, and label.
    """
    df = pd.read_csv(file_path, encoding = 'utf-8', sep = '\t')
    df = df[[df.columns[user_column], df.columns[post_column]]]
    df.columns = ['user_index', 'post']
    df['post'] = df['post'].astype(str)
    df['label'] = label;
    df = df[['user_index', 'post', 'label']]

    return df
    
def train(args, train_loader, val_loader):
    torch.cuda.empty_cache()
    model = BertForUserClassification(model_name=args.pre_trained, num_labels=args.num_classes)
    device = args.device
    model.to(device)
    optimizer = AdamW(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss()


    # Directory to save model checkpoints
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    num_epochs = args.num_epochs
    total_steps = len(train_loader) * num_epochs
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    accumulation_steps = args.num_accum_steps  # Adjust as needed
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        # Add tqdm for the training loop
        progress_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader), leave=True)
        
        for i, (batch_user_indices, inputs, labels) in progress_bar:
            if len(inputs['input_ids']) == 0:
                # Skip this batch if all inputs are empty
                continue
    
            inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
            labels = labels.to(device)
    
            # Apply mask to filter out dummy indices (-1)
            mask = (labels != -1)
            
            # Filter out dummy indices in inputs
            valid_indices = mask.nonzero(as_tuple=True)[0]
            input_ids = inputs['input_ids'][valid_indices]
            attention_mask = inputs['attention_mask'][valid_indices]
            if 'token_type_ids' in inputs:
                token_type_ids = inputs['token_type_ids'][valid_indices]
            else:
                token_type_ids = None
    
            # Ensure the tensors are on the correct device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            valid_labels = labels[valid_indices].to(device)
    
            # Pass only valid inputs to the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    
            # Compute the loss
            loss = criterion(outputs, valid_labels)
            loss = loss / accumulation_steps
            loss.backward()
            total_loss += loss.item()
    
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update the progress bar description with the current loss
            progress_bar.set_postfix(loss=loss.item())
    
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')
        # Save the model at the end of each epoch
        args.checkpoint_path = os.path.join(args.checkpoint_dir, f'BERT_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), args.checkpoint_path)
        print(f'Saved model checkpoint at {args.checkpoint_path}')

        evaluate(args, val_loader)

def evaluate(args, val_loader):
    model = BertForUserClassification(model_name=args.pre_trained, num_labels=args.num_classes)
    state_dict = torch.load(args.checkpoint_path)
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    total_loss = 0
    user_predictions = {}
    user_true_labels = {}

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating", leave=True)
        
        for batch_user_indices, inputs, labels in progress_bar:
            # Apply mask to filter out dummy indices (-1)
            mask = (labels != -1)
            valid_indices = mask.nonzero(as_tuple=True)[0]

            # Filter out dummy indices in inputs
            input_ids = inputs['input_ids'][valid_indices]
            attention_mask = inputs['attention_mask'][valid_indices]
            if 'token_type_ids' in inputs:
                token_type_ids = inputs['token_type_ids'][valid_indices]
            else:
                token_type_ids = None

            # Ensure the tensors are on the correct device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            valid_labels = labels[valid_indices].to(device)

            # Pass only valid inputs to the model
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            #logits = outputs.logits
            
            # Compute loss
            loss = criterion(logits, valid_labels)
            total_loss += loss.item()

            # Collect predictions and true labels for each user
            for user_index, logit, label in zip(batch_user_indices, logits, valid_labels):
                if user_index == -1:
                    continue  # Skip dummy indices
                if user_index not in user_predictions:
                    user_predictions[user_index] = []
                    user_true_labels[user_index] = label.item()
                user_predictions[user_index].append(logit.cpu().numpy())

    # Aggregate predictions for each user
    user_final_predictions = {}
    for user_index, preds in user_predictions.items():
        # Here we use the mean of the logits as the final prediction
        avg_preds = torch.tensor(preds).mean(dim=0)
        user_final_predictions[user_index] = torch.argmax(avg_preds).item()

    # Compute accuracy
    correct = 0
    total = len(user_final_predictions)
    for user_index, pred in user_final_predictions.items():
        if pred == user_true_labels[user_index]:
            correct += 1
    accuracy = correct / total

    avg_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss}, Validation Accuracy: {accuracy}')
    return avg_loss, accuracy
