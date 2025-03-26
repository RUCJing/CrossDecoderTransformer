# -*- coding: utf-8 -*-
import os

# files
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE_PATH = os.path.join(PROJECT_DIR, './dataset/train_df_cdt.feather')
TEST_FILE_PATH = os.path.join(PROJECT_DIR, './dataset/val_df_cdt.feather')
EVAL_FILE_PATH = os.path.join(PROJECT_DIR, './dataset/test_df_cdt.feather')
MODEL_FOLDER = os.path.join(PROJECT_DIR, './models/')
MODEL_PATH = os.path.join(PROJECT_DIR, './models/model_19.pth')

# for preprocessing
NUM_WORDS = 6000
PAD_SPEAKER = 2
PAD_TURN = 200
PAD_NO = 0
UNK_NO = 1
START_NO = UNK_NO + 1
SENT_LENGTH = 800

# hyperparameters
EMBEDDING_SIZE = 300
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 16
LEARNING_RATE = 0.0006
EPOCHS = 30

# models
ABLATION1 = False
ABLATION3 = False
