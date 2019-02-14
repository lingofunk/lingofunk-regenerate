import sys
import os
from pathlib import Path

PROJECT_FOLDER_PATH = Path(__file__).parent.parent

MODELS_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "models")
DATA_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "data")

DATA_FILE_NAME_DEFAULT = "review.csv"
DATA_FILE_PATH_DEFAULT = os.path.join(DATA_FOLDER_PATH, DATA_FILE_NAME_DEFAULT)

PORT_DEFAULT = 8000

BATCH_SIZE = 32
H_DIM = 64
Z_DIM = H_DIM  # 20
C_DIM = 2

LR = 2e-3
DROPOUT = 0.1

NUM_ITERATIONS_LR_DECAY = 1000000
NUM_ITERATIONS_TOTAL = 10000
NUM_ITERATIONS_LOG = 1000

MBSIZE = 32
KL_WEIGHT_MAX = 0.4

BETA = 0.1
LAMBDA_C = 0.1
LAMBDA_Z = 0.1
LAMBDA_U = 0.1
