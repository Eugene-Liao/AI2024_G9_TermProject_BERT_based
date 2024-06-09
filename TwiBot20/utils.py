import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import json
import torch
import torch.nn.functional as F
import torch.nn as nn

from models import  BertForUserClassification, CombinedModel
from tqdm import tqdm
from tabulate import tabulate
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to remove emojis
def remove_emojis(text):
    """
    Remove emojis from a given text.

    Args:
    text (str): Input text from which emojis need to be removed.

    Returns:
    str: Text with emojis removed.
    """
    if not isinstance(text, str):
        return text
    
    emoji_pattern = re.compile(
        "["
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U00002600-\U000026FF"  # Miscellaneous Symbols
        u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
        "]+", re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

# Function to remove URLs
def remove_urls(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'http\S+|www.\S+', '', text)


def tweetProcess(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame(data)
    df = df[['ID', 'tweet', 'label']]

    #explode the 'tweet'column
    df = df.explode('tweet')

    #filter-out non-string tweets
    df = df[df['tweet'].apply(lambda x: isinstance(x, str))]
    
    #remove url, emoji
    df['tweet']= df['tweet'].apply(remove_emojis).apply(remove_urls)

    #rename the columns
    df = df.rename(columns={"ID": "user_index", "tweet": "post", "label": "label"})

    #reset index for better reliability
    df.reset_index(drop=True, inplace=True)

    return df

def get_bert_embeddings(texts, tokenizer, model, device, max_length, batch_size):
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting BERT embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
  
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                token_type_ids=inputs.get('token_type_ids'), 
                return_embeddings=True
            )
    
        # outputs is already the logits from the [CLS] token, but we want the pooled output before classification
        pooled_output = outputs.detach().cpu().numpy()
        embeddings.extend(pooled_output)
    
    return embeddings


def postMetaDataPreProcess(data, tokenizer, model, device, max_length, batch_size):
    df = pd.DataFrame(data)
    df = df[['ID', 'tweet', 'profile', 'label']]
    df = df.rename(columns={"ID": "user_index", "tweet": "post", "profile": "profile", "label": "label"})

    #explode the 'tweet'column
    df = df.explode('post')
    #filter-out non-string tweets
    df = df[df['post'].apply(lambda x: isinstance(x, str))]
    #remove url, emoji
    df['post']= df['post'].apply(remove_emojis).apply(remove_urls)

    #get post embeddings, Apply BERT embeddings to the 'post' column with progress bar
    #tqdm.pandas(desc="Applying BERT embeddings")
    posts = df['post'].tolist()
    df['embeddings'] = get_bert_embeddings(posts, tokenizer, model, device, max_length, batch_size)
    
    # Extract 'verified' 'protected' status and convert 'True'/'False' to 1/0
    df['verified'] = df['profile'].apply(lambda x: 1 if x.get('verified') == 'True' else 0)
    df['protected'] = df['profile'].apply(lambda x: 1 if x.get('protected') == 'True' else 0)

    df['followers_count'] = df['profile'].apply(lambda x: x.get('followers_count', 0))
    df['friends_count'] = df['profile'].apply(lambda x: x.get('friends_count', 0))
    df['favorites_count'] = df['profile'].apply(lambda x: x.get('favorites', 0))
    df['statuses_count'] = df['profile'].apply(lambda x: x.get('statuses_count', 0))


    df = df.drop(columns = ['profile', 'post'])
    
    #reset index for better reliability
    df.reset_index(drop=True, inplace=True)

    return df



def train(args, train_loader, val_loader):
    torch.cuda.empty_cache()
    model = BertForUserClassification(model_name=args.pre_trained, num_labels=args.num_classes)
    device = torch.device(args.device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss()


    # Directory to save model checkpoints
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    num_epochs = args.num_epochs
    total_steps = len(train_loader) * num_epochs
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    accumulation_steps = args.accumulation_steps  # Adjust as needed
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

def combineTrain(args, train_loader, val_loader):
    torch.cuda.empty_cache()
    model = CombinedModel(args.bert_embedding_dim, args.metadata_dim, args.hidden_dim, args.output_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    device = torch.device(args.device)
    model.to(device)

    # Directory to save model checkpoints
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_epochs = args.num_epochs
    total_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    accumulation_steps = args.accumulation_steps  # Adjust as needed

    #training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
    
        # Add tqdm for the training loop
        progress_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader), leave=True)
    
        for i, (batch_user_indices, batch_labels, batch_embeddings, batch_numerical_data) in progress_bar:
            if len(batch_embeddings) == 0:
                # Skip this batch if all inputs are empty
                continue

            metadata = batch_numerical_data.to(device)
            embeddings = batch_embeddings.to(device)
            labels = batch_labels.to(device)

            # Apply mask to filter out dummy indices (-1)
            mask = (labels != -1)
        
            # Filter out dummy indices in inputs
            valid_indices = mask.nonzero(as_tuple=True)[0]
        
            valid_embeddings = embeddings[valid_indices].to(device)
            valid_metadata = metadata[valid_indices].to(device)
            valid_labels = labels[valid_indices].to(device).float()

            # Pass only valid inputs to the model
            outputs = model(valid_embeddings, valid_metadata)

            # Compute the loss
            loss = criterion(outputs.squeeze(), valid_labels)
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
        checkpoint_path = os.path.join(checkpoint_dir, f'BERT_tweets_combine_256_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved model checkpoint at {checkpoint_path}')

        combinEvaluate(args, val_loader)

def evaluate(args, val_loader):
    model = BertForUserClassification(model_name=args.pre_trained, num_labels=args.num_classes)
    state_dict = torch.load(args.checkpoint_path)
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()
    device = torch.device(args.device)
 
    model.to(device)
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

            if len(input_ids) == 0:
               # Skip this batch if all inputs are empty
               continue

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

    # Compute metrics
    true_labels = []
    pred_labels = []

    for user_index, pred in user_final_predictions.items():
        true_labels.append(user_true_labels[user_index])
        pred_labels.append(pred)

    accuracy = sum([1 for true, pred in zip(true_labels, pred_labels) if true == pred]) / len(true_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    avg_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss}, Validation Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    return avg_loss, accuracy, precision, recall, f1

def combinEvaluation(args, val_loader):
    model = CombinedModel(args.bert_embedding_dim, args.metadata_dim, args.hidden_dim, args.output_dim)
    state_dict = torch.load(args.checkpoint_path)
    model.load_state_dict(state_dict)
    
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    total_loss = 0
    user_predictions = {}
    user_true_labels = {}

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating", leave=True)
        
        for batch_user_indices, batch_labels, batch_embeddings, batch_numerical_data in progress_bar:
            # Apply mask to filter out dummy indices (-1)
            mask = (batch_labels != -1)
            valid_indices = mask.nonzero(as_tuple=True)[0]
            
            valid_indices = valid_indices.to('cpu')
            valid_embeddings = batch_embeddings[valid_indices].to(device)
            valid_metadata = batch_numerical_data[valid_indices].to(device)
            valid_labels = batch_labels[valid_indices].to(device).float()

            # Pass only valid inputs to the model
            logits = model(valid_embeddings, valid_metadata)
                        
            # Compute loss
            loss = criterion(logits.squeeze(), valid_labels)
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
        user_final_predictions[user_index] = torch.sigmoid(avg_preds).item()

    # Compute metrics
    true_labels = []
    pred_labels = []

    threshold = 0.5
    for user_index, pred in user_final_predictions.items():
        true_labels.append(user_true_labels[user_index])
        pred_labels.append(1 if pred >= threshold else 0)

    accuracy = sum([1 for true, pred in zip(true_labels, pred_labels) if true == pred]) / len(true_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    avg_loss = total_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss}, Validation Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    return avg_loss

