# BERT-based model for TwiBot-20 Dataset
BERT is fine-tuned for fake account detection based on the TwiBot-20 dataset. Additional user meta-data was added to enhance classification performance based on the fine-tuned BERT model. However, the data set was not publicaly available due to privacy issue. 

## Data Formats
Each user contains:
- 'ID': the ID from Twitter identifying the user.
- 'profile': the profile information obtained from Twitter API.
- 'tweet': the latest 200 tweets of this user.
- 'neighbor': random 20 followers and followings of this user.
- 'domain': the domain of this user (domains: politics, business, entertainment and sports).
- 'label': '1' represents it is a bot and '0' represents it is a human.

## Usage
- `$python3 twiBot20_train_eval.py` fine-tune BERT
- `$python3 twiBot20_combine_train_eval.py` train with additional meta-data with the embedding of the fine-tuned BERT
- `$python3 twiBot20_evaluation.py` evaluatioon for the fine-tuned BERT
- `$python3 twiBot20_combine_eval.py` evaluation for the combined moodel

## Affilicated files
- `utils.py`: data pre-processing, `train()` and `evaluate()`
- `models.py`: model class
- `Data.py`: modified Dataset class, Sampler and custom_collate()
