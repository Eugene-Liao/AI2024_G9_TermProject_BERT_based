# BERT-based model for Weibo Dataset
BERT is fine-tuned for fake account detection based on [this shared dataset](https://www.kaggle.com/datasets/bitandatom/social-network-fake-account-dataset
).

## File Description

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
-[Utils.py]
utils.py: data pre-processing, train() and evaluate()
models.py: model class
Data.py: modified Dataset class, Sampler and custom_collate()


Evaluation: $python3 evaluation.py
-- during evaluation, the predicted logits belonging to the same user is averaged to make final classificatioon


Data Formats:
fake_account.csv: userindex \t post_content \n
legitimate_account.csv: userindex \t postdate \t retweet_count \t comment_count \t like_count \t post_content \n
