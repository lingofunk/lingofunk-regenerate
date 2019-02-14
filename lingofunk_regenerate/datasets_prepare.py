import argparse
import csv
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))


from lingofunk_regenerate.constants import (
    DATA_FOLDER_PATH,
    NROWS_TO_READ_FROM_CSV,
    MAX_NUM_WORDS_TO_FILTER,
    EMB_DIM
)

from lingofunk_regenerate.datasets import YelpDataset as Dataset

from lingofunk_regenerate.utils import log as _log


def log(text):
    _log(text, prefix='Datasets prepare: ')


def prepare_yelp_data(data_folder=DATA_FOLDER_PATH,
                      data_file="reviews.csv",
                      train="reviews_train.csv",
                      test="reviews_test.csv",
                      val="reviews_val.csv",
                      cols=["text", "stars"]):
    data_path = os.path.join(data_folder, data_file)
    index_col = 0

    df = pd.read_csv(data_path, usecols=cols, sep=",", index_col=index_col, nrows=NROWS_TO_READ_FROM_CSV)
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

        words_splitted = [words[i : i + MAX_NUM_WORDS_TO_FILTER] for i in range(0, len(words) - MAX_NUM_WORDS_TO_FILTER, MAX_NUM_WORDS_TO_FILTER)]
        labels_splitted = len(words_splitted) * [label]

        labels_expanded += labels_splitted
        texts_expanded += [" ".join(ws) for ws in words_splitted]

    df_new = pd.DataFrame()
    df_new["label"] = labels_expanded
    df_new["text"] = texts_expanded

    df_train, df_test = train_test_split(df_new, test_size=0.2)
    df_train, df_val = train_test_split(df_train, test_size=0.1)

    print(df_train.head())

    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    df_val.reset_index(inplace=True, drop=True)

    print(df_train.shape)
    print(df_test.shape)
    print(df_val.shape)

    print(df_train.head())

    df_train.to_csv(
        os.path.join(data_folder, train),
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    df_test.to_csv(
        os.path.join(data_folder, test),
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    df_val.to_csv(
        os.path.join(data_folder, val),
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
    )


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-folder",
        type=str,
        required=True,
        help="Data folder",
    )

    parser.add_argument(
        "--data-file",
        type=str,
        required=False,
        help="Data file",
    )

    parser.add_argument(
        "--train",
        type=str,
        required=False,
        help="Train file name",
    )

    parser.add_argument(
        "--test",
        type=str,
        required=False,
        help="Test file name",
    )

    parser.add_argument(
        "--val",
        type=str,
        required=False,
        help="Val file name",
    )

    parser.add_argument(
        "--save-train-test-val",
        action="store_true",
        required=False,
        help="Whether to create and save train, test and val files",
    )

    parser.add_argument(
        "--save-fields",
        action="store_true",
        required=False,
        help="Whether to save torchtext fields or not",
    )

    parser.add_argument(
        "--emb-dim",
        type=int,
        required=False,
        default=EMB_DIM,
        help="Embedding dim",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.save_train_test_val:
        log('prepare data')
        prepare_yelp_data(
            args.data_folder, args.data_file,
            args.train, args.test, args.val
        )

    if args.save_fields:
        log('initialize dataset')
        dataset = Dataset(
            emb_dim=args.emb_dim,
            data_folder=args.data_folder,
            train=args.train,
            test=args.test,
            val=args.val,
            init_from_data=True)

        log('save fields')
        dataset.save_fields(args.data_folder)
