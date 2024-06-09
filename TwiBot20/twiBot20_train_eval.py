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
    #load and preprocess
    train_df = utils.tweetProcess(args.train_path)
    val_df = utils.tweetProcess(args.val_path)
        
    tokenizer = BertTokenizer.from_pretrained(args.pre_trained)

    # Tokenizing, batching, loading
    train_set = Data.SocialMediaDataset(train_df, tokenizer, max_length=512)
    val_set = Data.SocialMediaDataset(val_df, tokenizer, max_length=512)

    train_batch_sampler = Data.UserBatchSampler(train_set, args.batch_size)
    val_batch_sampler = Data.UserBatchSampler(val_set, args.batch_size)

    train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, collate_fn=Data.custom_collate(tokenizer))
    val_loader = DataLoader(val_set, batch_sampler=val_batch_sampler, collate_fn=Data.custom_collate(tokenizer))
    
    utils.train(args, train_loader, val_loader)


def parse_args() -> Namespace:
    parser = ArgumentParser()

    #data
    parser.add_argument("--train_path", type=str, default =  "./data/Twibot-20/train.json", help='train file path') 
    parser.add_argument("--val_path", type=str, default='. "./data/Twibot-20/test.json"', help='test file path')
    
    #model setting, hyperparameters
    parser.add_argument("--pre_trained", type=str, default='bert-base-uncased', help="Name of the pre-trained model (e.g., 'bert-base-uncased').")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes." )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (cuda or cpu).")
    
    #result saving
    parser.add_argument("--checkpoint_dir", type=str, default="./test_checkpoints", help="Dir to save the model checkpoints.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to the model checkpoints.")
    
    
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = parse_args()
    main(args)