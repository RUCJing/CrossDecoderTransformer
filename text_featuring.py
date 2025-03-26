# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch as T
from torch.utils.data import Dataset, random_split

from params import (PAD_NO,
                    UNK_NO,
                    START_NO,
                    SENT_LENGTH,
                   PAD_SPEAKER)
from pickle_file_operaor import PickleFileOperator


# load csv file
def load_csv_file(file_path):
    df = pd.read_feather(file_path)
    X_cat = df[['id', 'department', 'title']].values
    X_num = np.vstack(df['quant'])
    y = df['label'].values
    samples, speakers = [], []
    for index, row in df.iterrows():
        samples.append(row['Text'])
        speakers.append(row['speaker'])
    return samples, y, speakers, X_cat, X_num


# 读取pickle文件
def load_file_file():
    labels = PickleFileOperator(file_path='labels.pk').read()
    chars = PickleFileOperator(file_path='chars.pk').read()
    label_dict = dict(zip(labels, range(len(labels))))
    char_dict = dict(zip(chars, range(len(chars))))
    return label_dict, char_dict


# 文本预处理
def text_feature(labels, contents, speakers, label_dict, char_dict):
    samples, y_true, s_speakers = [], [], []
    for s_label, s_content, s_speaker in zip(labels, contents, speakers):
        s_speaker = list(s_speaker)
        y_true.append(label_dict[s_label])
        train_sample = []
        for char in s_content:
            if char in char_dict:
                train_sample.append(START_NO + char_dict[char])
            else:
                train_sample.append(UNK_NO)
        if len(train_sample) < SENT_LENGTH:
            samples.append(train_sample + ([PAD_NO] * (SENT_LENGTH - len(train_sample))))
            s_speakers.append(s_speaker + ([PAD_SPEAKER] * (SENT_LENGTH - len(train_sample))))
        else:
            samples.append(train_sample[:SENT_LENGTH])
            s_speakers.append(s_speaker[:SENT_LENGTH])

    return samples, y_true, s_speakers


# dataset
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, file_path):
        label_dict, char_dict = load_file_file()
        text, y, speakers, X_cat, X_num = load_csv_file(file_path)
        text, y, speakers = text_feature(y, text, speakers, label_dict, char_dict)
        self.text = T.from_numpy(np.array(text)).long()
        self.speakers = T.from_numpy(np.array(speakers)).long()
        self.y = T.from_numpy(np.array(y))
        self.X_cat = T.from_numpy(X_cat)
        self.X_num = T.from_numpy(X_num)

    # number of rows in the dataset
    def __len__(self):
        return len(self.y)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.text[idx], self.X_cat[idx], self.X_num[idx], self.y[idx], self.speakers[idx]]
    
    # get indexes for train and test rows
    def get_splits(self, n_test=0.3):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
