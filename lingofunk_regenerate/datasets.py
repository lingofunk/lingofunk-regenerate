import os
import sys
import torch
import dill
from torchtext import data, datasets
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

from lingofunk_regenerate.constants import (
    DATA_FOLDER_PATH,
    MAX_NUM_WORDS_TO_FILTER,
    EMB_DIM,
    BATCH_SIZE,
    GLOVE
)

# Saving/loading torch fields
# https://github.com/pytorch/text/pull/453/commits/064a204b1f715cca5e19af435a8f5519ee427aef


filter_func = lambda ex: len(ex.text) <= MAX_NUM_WORDS_TO_FILTER


def get_data_cb_sample(field_text, field_label):
    train, val, test = datasets.SST.splits(
        field_text,
        field_label,
        fine_grained=False,
        train_subtrees=False,
        filter_pred=filter_func,
    )

    return train, val, test


class Dataset:
    def __init__(self,
                 emb_dim=EMB_DIM,
                 mbsize=BATCH_SIZE,
                 get_data_cb=get_data_cb_sample,
                 init_from_data=False):

        self.TEXT = None
        self.LABEL = None
        self.n_vocab = None

        self.emb_dim = emb_dim
        self.mbsize = mbsize
        self.get_data_cb = get_data_cb

        self.train_iter, self.val_iter = None, None

        if init_from_data:
            self.initialize_from_data()

    def initialize_from_data(self):
        self.TEXT = data.Field(
            init_token="<start>",
            eos_token="<eos>",
            lower=True,
            tokenize="spacy",
            fix_length=MAX_NUM_WORDS_TO_FILTER + 1,
        )
        self.LABEL = data.Field(
            sequential=False,
            unk_token=None
        )

        train, val, test = self.get_data_cb(self.TEXT, self.LABEL)

        self.TEXT.build_vocab(
            train,
            vectors=GloVe(
                GLOVE,
                dim=self.emb_dim,
                cache=DATA_FOLDER_PATH)
        )
        self.LABEL.build_vocab(train)

        # print(self.TEXT.vocab)
        # print(self.LABEL.vocab)

        self.n_vocab = len(self.TEXT.vocab.itos)

        self.train_iter, self.val_iter, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=self.mbsize, shuffle=True, repeat=True
        )
        self.train_iter = iter(self.train_iter)
        self.val_iter = iter(self.val_iter)

    def load_fields(self, folder_path=None):
        if folder_path is None:
            folder_path = self.data_folder

        text_pickle_path = os.path.join(folder_path, 'text.pl')
        label_pickle_path = os.path.join(folder_path, 'label.pl')

        # self.TEXT = torch.load(text_pickle_path)
        # self.LABEL = torch.load(label_pickle_path)

        with open(text_pickle_path, "rb") as f:
            self.TEXT = dill.load(f)

        with open(label_pickle_path, "rb") as f:
            self.LABEL = dill.load(f)

        self.n_vocab = len(self.TEXT.vocab.itos)

    def save_fields(self, folder_path=None):
        if folder_path is None:
            folder_path = self.data_folder

        text_pickle_path = os.path.join(folder_path, 'text.pl')
        label_pickle_path = os.path.join(folder_path, 'label.pl')

        # torch.save(self.TEXT, text_pickle_path)
        # torch.save(self.LABEL, label_pickle_path)

        with open(text_pickle_path, "wb") as f:
            dill.dump(self.TEXT, f)

        with open(label_pickle_path, "wb") as f:
            dill.dump(self.LABEL, f)

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def next_batch(self, gpu=False):
        batch = next(self.train_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        # print(batch.text)

        return batch.text, batch.label

    def next_validation_batch(self, gpu=False):
        batch = next(self.val_iter)

        if gpu:
            return batch.text.cuda(), batch.label.cuda()

        return batch.text, batch.label

    def idxs2sentence(self, idxs):
        return " ".join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]

    def word2idx(self, word):
        return self.TEXT.vocab.stoi[word]

    def sentence2idxs(self, sentence):
        return [self.word2idx(s) for s in sentence.split(" ")]


# HAIKU

## Prepare
# import os
# import pandas as pd
# import re
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# DATA_FOLDER_PATH = os.path.join(
#     '/media/student/8d1913cf-1155-47a5-a7db-b9a51f445d8f/student/data/hAIku')
#
# data_path = os.path.join(
#     DATA_FOLDER_PATH, 'data.tsv')
#
# nrows = 50000
#
# df = pd.read_csv(data_path, sep='\t', nrows=nrows)
#
# print(df.head())
# print(df.shape)
#
# df['label'] = 1
#
# df.columns = ['text', 'labels']
#
# texts = df['text'].values
# lens = [len(t.split(' ')) for t in texts]
#
# print(texts[:5])
# print(lens[:5])
# print('Max len "{}"'.format(max(lens)))
# print('Mid len "{}"'.format(np.quantile(lens, 0.5)))
#
# print(df.head())
#
# df['text'] = df['text'].apply(lambda t: re.sub('>', '> ', re.sub('<', ' <', t)))
# df['text'] = df['text'].apply(lambda t: ' '.join(t.split('\n')))
#
# df_train, df_test = train_test_split(df, test_size=0.2)
# df_train, df_val = train_test_split(df_train, test_size=0.1)
#
# print(df_train.head())
#
# df_train.reset_index(inplace=True, drop=True)
# df_test.reset_index(inplace=True, drop=True)
# df_val.reset_index(inplace=True, drop=True)
#
# print(df_train.shape)
# print(df_test.shape)
# print(df_val.shape)
#
# print(df_train.head())
#
# df_train.to_csv(os.path.join(DATA_FOLDER_PATH, 'train.csv'), index=False)
# df_test.to_csv(os.path.join(DATA_FOLDER_PATH, 'test.csv'), index=False)
# df_val.to_csv(os.path.join(DATA_FOLDER_PATH, 'val.csv'), index=False)
#
#
# data_path = os.path.join(
#     '/media/student/8d1913cf-1155-47a5-a7db-b9a51f445d8f/student/data/hAIku/data.tsv')
#
# df = pd.read_csv(data_path, sep='\t')
#
# print(df.head())
#
#
# ## Dataset
#
# import os
# from torchtext.data import Field
# from torchtext.data import TabularDataset
#
# from .dataset import Dataset
#
# tokenize = lambda x: x.split(' ')
#
# TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
# LABEL = Field(sequential=False, use_vocab=False)
#
# DATA_FOLDER_PATH = os.path.join(
#     '/media/student/8d1913cf-1155-47a5-a7db-b9a51f445d8f/student/data/hAIku')
# cols = ['text', 'stars']
#
# # print(all_datafields)
#
# def get_haiku_data(field_text, field_label):
#     train, val, test = TabularDataset.splits(
#         path=DATA_FOLDER_PATH,
#         train='train.csv',
#         validation='val.csv',
#         test='test.csv',
#         format='csv',
#         skip_header=True,
#         fields=[('text', field_text), ('label', field_label)])
#
#     return train, val, test
#
# # tst_datafields = [('text', TEXT)]
# #
# # tst = TabularDataset(
# #     path=DATA_FOLDER_PATH,  # the file path
# #     format='csv',
# #     skip_header=True,
# #     # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
# #     fields=tst_datafields)
#
# class HaikuDataset(Dataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs, get_data_cb=get_haiku_data)
#


def get_yelp_data(field_text, field_label,
                  data_folder, train, test, val):
    train, val, test = TabularDataset.splits(
        path=data_folder,
        train=train,
        validation=val,
        test=test,
        format="csv",
        skip_header=True,
        fields=[("label", field_label), ("text", field_text)],
    )

    return train, val, test


class YelpDataset(Dataset):
    def __init__(self,
                 data_folder=DATA_FOLDER_PATH,
                 train="reviews_train.csv",
                 test="reviews_val.csv",
                 val="reviews_test.csv",
                 *args, **kwargs):

        def get_yelp_data_cb(field_text, field_label):
            return get_yelp_data(field_text, field_label, data_folder, train, test, val)

        super().__init__(*args, **kwargs, get_data_cb=get_yelp_data_cb)
        self.data_folder = data_folder
