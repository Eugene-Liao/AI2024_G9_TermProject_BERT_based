import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils, Data

from models import  BertForUserClassification
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from tabulate import tabulate
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


def main(args):
    fake_df = utils.dataProcess_simple(args.fake_account_file, 0, 1, 0)
    legitimate_df = utils.dataProcess_simple(args.legal_account_file, 0, 5, 1)

    # Combine both DataFrames
    combined_df = pd.concat([fake_df, legitimate_df], ignore_index=True)

    # Shuffle the combined DataFrame
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    tokenizer = BertTokenizer.from_pretrained(args.pre_trained)

    train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)

    # Assuming train_df and val_df are your training and validation dataframes
    train_set = Data.SocialMediaDataset(train_df, tokenizer)
    val_set = Data.SocialMediaDataset(val_df, tokenizer)

    train_batch_sampler = Data.UserBatchSampler(train_set, args.batch_size)
    val_batch_sampler = Data.UserBatchSampler(val_set, args.batch_size)

    train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, collate_fn=Data.custom_collate)
    val_loader = DataLoader(val_set, batch_sampler=val_batch_sampler, collate_fn=Data.custom_collate)
    
    train(args, train_loaoder)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()

    #data
    parser.add_argument("--fake_account_file", type=str, default = './fake_account.csv', help='fake account file path') 
    parser.add_argument("--legitimate_account_file", type=str, default='./legitimate_account.csv', help='legitimate account file path')

    #model setting
    parser.add_argument("--pre_trained", type=str, help="Name of the pre-trained model (e.g., 'bert-base-uncased').", default='bert-base-chinese')
    parser.add_argument("--num_classes", type=int, required=True, default=2, help="Number of output classes." )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Dir to save the model checkpoints.")
    
    args.parser.parse_args()

    return args



if __name__ == '__main__':
    args = parse_args()
    main(args)