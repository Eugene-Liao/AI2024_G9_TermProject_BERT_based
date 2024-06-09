# BERT-based model for TwiBot-20 Dataset
BERT is fine-tuned for fake account detection based on the TwiBot-20 dataset. Additional user meta-data was added to enhance classification performance based on the fine-tuned BERT model. However, the data set was not publicaly available due to privacy issue. 

## Data Formats
- fake_account.csv: userindex \t post_content \n
- legitimate_account.csv: userindex \t postdate \t retweet_count \t comment_count \t like_count \t post_content \n

## Usage
- `$python3 twiBot20_train_eval.py` fine-tune BERT
- `$python3 twiBot20_combine_train_eval.py` train with additional meta-data with the embedding of the fine-tuned BERT
- `$python3 twiBot20_evaluation.py` evaluatioon for the fine-tuned BERT
- `$python3 twiBot20_combine_eval.py` evaluation for the combined moodel

## Arguments


## Affilicated files
- `utils.py`: data pre-processing, `train()` and `evaluate()`
- `models.py`: model class
- `Data.py`: modified Dataset class, Sampler and custom_collate()
