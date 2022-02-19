import torch
import torchtext.legacy.data
from torchtext import data, datasets
import random

TEXT = torchtext.legacy.data.Field(tokenize='spacy', lower=True, tokenizer_language='en_core_web_sm')
LABEL = torchtext.legacy.data.LabelField(dtype=torch.float)

train_data, test_data = torchtext.legacy.datasets.IMDB.splits(text_field=TEXT, label_field=LABEL)

train_data, valid_data = train_data.split(random_state=random.seed(1234))

TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, valid_iter, test_iter = torchtext.legacy.data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                                batch_size=batch_size, device=device)
