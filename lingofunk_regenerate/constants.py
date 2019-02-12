import sys
import os

project_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, project_folder)

MODELS_FOLDER_PATH = os.path.join(project_folder, 'models')
DATA_FOLDER_PATH = os.path.join(project_folder, 'data')

DATA_FILE_NAME_DEFAULT = 'review.csv'
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
