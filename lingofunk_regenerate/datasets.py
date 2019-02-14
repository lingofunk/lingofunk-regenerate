import csv
import os

import pandas as pd
from lingofunk_regenerate.constants import DATA_FOLDER_PATH
from sklearn.model_selection import train_test_split
from torchtext import data, datasets
from torchtext.data import Field, TabularDataset
from torchtext.vocab import GloVe

# 15
filter_func = lambda ex: len(ex.text) <= 30 and ex.label != "neutral"


def get_sst_data(field_text, field_label):
    train, val, test = datasets.SST.splits(
        field_text,
        field_label,
        fine_grained=False,
        train_subtrees=False,
        filter_pred=filter_func,
    )

    return train, val, test


class Dataset:
    def __init__(self, emb_dim=50, mbsize=32, get_data_cb=get_sst_data):
        self.TEXT = data.Field(
            init_token="<start>",
            eos_token="<eos>",
            lower=True,
            tokenize="spacy",
            fix_length=16,
        )
        self.LABEL = data.Field(sequential=False, unk_token=None)

        train, val, test = get_data_cb(self.TEXT, self.LABEL)

        self.TEXT.build_vocab(
            train, vectors=GloVe("6B", dim=emb_dim, cache=DATA_FOLDER_PATH)
        )
        self.LABEL.build_vocab(train)

        # print(self.TEXT.vocab)
        # print(self.LABEL.vocab)

        self.n_vocab = len(self.TEXT.vocab.itos)
        self.emb_dim = emb_dim

        self.train_iter, self.val_iter, _ = data.BucketIterator.splits(
            (train, val, test), batch_size=mbsize, shuffle=True, repeat=True
        )
        self.train_iter = iter(self.train_iter)
        self.val_iter = iter(self.val_iter)

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


# Yelp

## Prepare


def prepare_yelp_data():
    data_path = os.path.join(DATA_FOLDER_PATH, "reviews.csv")
    # data_path_txt = os.path.join(DATA_FOLDER_PATH, 'reviews.txt')
    cols = ["text", "stars"]
    index_col = 0
    nrows = 500000

    df = pd.read_csv(data_path, usecols=cols, sep=",", index_col=index_col, nrows=nrows)
    df.reset_index(inplace=True)

    # print(df.shape)

    df.rename(columns={"stars": "label"}, inplace=True)
    df.drop(index=df[df["label"] == 3.0].index, axis=0, inplace=True)

    # print(df.shape)

    labels = df["label"].values
    labels[labels == 2] = 0
    labels[labels == 1] = 0
    labels[labels == 5] = 0
    labels[labels == 4] = 0

    df["label"] = labels

    texts = df["text"].values

    for i in range(len(texts)):
        texts[i] = " ".join(texts[i].split("\n"))

    # with open(data_path_txt, 'w') as f:
    #     f.write('\n'.join(texts))

    labels_expanded = []
    texts_expanded = []

    for text, label in zip(texts, labels):
        text = " ".join(text.split("\n"))
        words = text.split(" ")
        words_splitted = [words[i : i + 15] for i in range(0, len(words) - 15, 15)]
        labels_splitted = len(words_splitted) * [label]

        # print(words)
        # print(words_splitted)
        # print(labels_splitted)
        # print(len(words))
        # print(len(words_splitted))
        # print(len(labels_splitted))

        labels_expanded += labels_splitted
        texts_expanded += [" ".join(ws) for ws in words_splitted]

    df_new = pd.DataFrame()
    df_new["label"] = labels_expanded
    df_new["text"] = texts_expanded

    df_train, df_test = train_test_split(df_new, test_size=0.2)
    df_train, df_val = train_test_split(df_train, test_size=0.1)

    # df_train = df.sample(frac=0.8, random_state=200)
    # df_test = df.drop(df_train.index)
    #
    # df_train = df_train.sample(frac=0.9, random_state=200)
    # df_val = df_train.drop(df_train.index)

    print(df_train.head())

    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    df_val.reset_index(inplace=True, drop=True)

    print(df_train.shape)
    print(df_test.shape)
    print(df_val.shape)

    print(df_train.head())

    df_train.to_csv(
        os.path.join(DATA_FOLDER_PATH, "reviews_train.csv"),
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    df_test.to_csv(
        os.path.join(DATA_FOLDER_PATH, "reviews_test.csv"),
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    df_val.to_csv(
        os.path.join(DATA_FOLDER_PATH, "reviews_val.csv"),
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


## Dataset

tokenize = lambda x: x.split(" ")

TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = Field(sequential=False, use_vocab=False)

cols = ["text", "stars"]

all_datafields = [("stars", LABEL), ("text", TEXT)]

# print(all_datafields)


def get_yelp_data(field_text, field_label):
    train, val, test = TabularDataset.splits(
        path=DATA_FOLDER_PATH,
        train="reviews_train.csv",
        validation="reviews_val.csv",
        test="reviews_test.csv",
        format="csv",
        skip_header=True,
        fields=[("label", field_label), ("text", field_text)],
    )

    return train, val, test


# tst_datafields = [('text', TEXT)]
#
# tst = TabularDataset(
#     path=DATA_FOLDER_PATH,  # the file path
#     format='csv',
#     skip_header=True,
#     # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
#     fields=tst_datafields)


class YelpDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, get_data_cb=get_yelp_data)


if __name__ == "__main__":
    prepare_yelp_data()
