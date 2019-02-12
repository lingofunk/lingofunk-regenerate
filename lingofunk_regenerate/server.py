import sys
import os
import argparse
import logging
import time
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from lingofunk_regenerate.datasets import YelpDataset as Dataset
# from lingofunk_regenerate.datasets import HaikuDataset as Dataset
from lingofunk_regenerate.model import RNN_VAE
from lingofunk_regenerate.constants import BATCH_SIZE
from lingofunk_regenerate.constants import H_DIM
from lingofunk_regenerate.constants import Z_DIM
from lingofunk_regenerate.constants import C_DIM
from lingofunk_regenerate.constants import LR
from lingofunk_regenerate.constants import NUM_ITERATIONS_LR_DECAY
from lingofunk_regenerate.constants import NUM_ITERATIONS_TOTAL
from lingofunk_regenerate.constants import NUM_ITERATIONS_LOG
from lingofunk_regenerate.constants import DROPOUT
from lingofunk_regenerate.utils import log as _log
from lingofunk_regenerate import tests as tests

import random
import time


project_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, project_folder)


from lingofunk_regenerate.constants import MODELS_FOLDER_PATH
from lingofunk_regenerate.constants import PORT_DEFAULT


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)
graph = tf.get_default_graph()

model = None
dataset = None


def log(text):
    _log(text, prefix='Server: ')


@app.route('/hello', methods=['GET'])
def hello_world():
    return 'Hello World!'


@app.route('/regenerate', methods=['GET', 'POST'])
def generate_discrete():
    logger.debug('request: {}'.format(request.get_json()))

    global model
    global dataset

    # data = request.get_json()
    # text_style = data.get('style-name')

    sentence = request.args['text']
    angle = float(request.args['angle'])

    if angle > 180:
        angle = 360 - angle
        sign = -1
    else:
        angle = angle
        sign = +1

    radius = float(request.args['radius'])
    index = int(np.round(
        angle / 180 * (model.z_dim - 1)
    ))
    disturbance = sign * radius

    mbsize = 1

    sentence = dataset.sentence2idxs(sentence)
    sentence = np.array(sentence).reshape(-1, mbsize)
    sentence = torch.from_numpy(sentence)

    # Encoder: sentence -> z
    mu, logvar = model.forward_encoder(sentence)

    # if use_c_prior:
    c = model.sample_c_prior(mbsize)
    # else:
    #     c = model.forward_discriminator(sentence.transpose(0, 1))

    temp = 1.0

    regenerated_sentence_best = None
    num_hits_best = None

    for i in range(10):
        z = model.sample_z(mu, logvar)
        z[0, index] += disturbance * abs(z[0, index])

        sampled = model.sample_sentence(z, c, temp=temp)
        regenerated_sentence = dataset.idxs2sentence(sampled)
        num_hits = np.sum(sentence.numpy() == regenerated_sentence.numpy())

        if num_hits_best is None or num_hits > num_hits_best:
            num_hits_best = num_hits
            regenerated_sentence_best = regenerated_sentence

    log('Regenerated sentence: ' + regenerated_sentence_best + '\n')

    return jsonify(new_text=regenerated_sentence_best)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--port', type=int, required=False, default=PORT_DEFAULT,
        help='The port to listen on')

    parser.add_argument(
        '--seed', type=int, required=False, default=int(time.time()),
        help='Random seed')

    parser.add_argument(
        '--model', type=str, required=False, default='vae',
        choices=['vae', 'ctextgen'],
        help='Which model to use')

    parser.add_argument(
        '--models-folder', type=str, required=False, default=MODELS_FOLDER_PATH,
        help='Folder to load models (weights, vocabs, configs) from')

    parser.add_argument(
        '--gpu', required=False, action='store_true',
        help='whether to run in the GPU')

    return parser.parse_args()


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_model(model_name, models_folder, gpu=False):
    # TODO: pass to config

    global model
    global dataset

    dataset = Dataset()

    h_dim = H_DIM
    z_dim = Z_DIM
    c_dim = C_DIM
    p_word_dropout = DROPOUT

    model = RNN_VAE(
        dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=p_word_dropout,
        pretrained_embeddings=dataset.get_vocab_vectors(),
        freeze_embeddings=True,
        gpu=gpu)

    if gpu:
        model.load_state_dict(
            torch.load('{}/{}.bin'.format(models_folder, model_name)))
    else:  # TODO: DRY
        model.load_state_dict(
            torch.load('{}/{}.bin'.format(models_folder, model_name), map_location=lambda storage, loc: storage))


def _test_model(gpu=False):
    log('test model')

    global model
    global dataset

    tests.test_generate_text_of_sentiment(model, dataset)
    tests.test_interpolate(model, dataset, gpu)
    tests.test_encode_text_add_noize_and_decode(model, dataset)


def _main():
    args = _parse_args()

    _set_seed(args.seed)
    _load_model(args.model, args.models_folder, args.gpu)
    # _test_model(args.gpu)

    app.run(host='0.0.0.0', port=args.port, debug=True, threaded=True)


if __name__ == '__main__':
    _main()
