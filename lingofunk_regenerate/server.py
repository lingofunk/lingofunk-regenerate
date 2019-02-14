import argparse
import logging
import os
import re
import sys
import time
import torch

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

from lingofunk_regenerate import tests as tests
from lingofunk_regenerate.constants import (
    C_DIM,
    DROPOUT,
    H_DIM,
    MODELS_FOLDER_PATH,
    DATA_FOLDER_PATH,
    PORT_DEFAULT,
    Z_DIM,
)
from lingofunk_regenerate.datasets import YelpDataset as Dataset
from lingofunk_regenerate.model import RNN_VAE
from lingofunk_regenerate.utils import log as _log


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)
graph = tf.get_default_graph()

model = None
dataset = None


def log(text):
    _log(text, prefix="Server: ")


@app.route("/hello", methods=["GET"])
def hello_world():
    return "Hello World!"


@app.route("/regenerate", methods=["POST"])
def regenerate():
    logger.debug("request: {}".format(request.get_json()))

    global model
    global dataset

    data = request.get_json()

    sentence = data["text"]
    sentence = re.sub("\W", " ", sentence)

    logger.debug("Sentence: {}".format(sentence))

    angle = float(data["angle"])
    radius = float(data["radius"])
    disturbance_fraction = radius / 100

    if angle > 180:
        angle = 360 - angle
        sign = -1
    else:
        angle = angle
        sign = +1

    index = int(np.round(angle / 180 * (model.z_dim - 1)))
    disturbance_fraction_signed = sign * disturbance_fraction

    logger.debug("index to disturb: {}".format(index))
    logger.debug("disturbance fraction: {}".format(disturbance_fraction_signed))

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

    temp = 0.5

    regenerated_sentence = None
    num_hits_best = None

    for i in range(20):
        logger.debug("iter {}:".format(i))

        z = model.sample_z(mu, logvar)
        disturbance = disturbance_fraction_signed * abs(z[0, index])
        z[0, index] += disturbance

        sampled_idxs = model.sample_sentence(z, c, temp=temp)
        sampled_sentence = dataset.idxs2sentence(sampled_idxs)

        sampled_idxs = torch.from_numpy(np.array(sampled_idxs).reshape(-1, mbsize))

        num_hits = len(
            set(sentence.numpy().flatten()).intersection(
                set(sampled_idxs.numpy().flatten())
            )
        )

        logger.debug("sampled sentence: {}".format(sampled_sentence))
        logger.debug("num hits: {}".format(num_hits))

        if num_hits_best is None or num_hits > num_hits_best:
            num_hits_best = num_hits
            regenerated_sentence = sampled_sentence

    log("Regenerated sentence: " + regenerated_sentence + "\n")

    return jsonify({"new-text": regenerated_sentence})


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default=PORT_DEFAULT,
        help="The port to listen on",
    )

    parser.add_argument(
        "--seed", type=int, required=False, default=int(time.time()), help="Random seed"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="vae",
        help="Which model to use (vae, ctextgen or other *.bin file in models folder)",
    )

    parser.add_argument(
        "--models-folder",
        type=str,
        required=False,
        default=MODELS_FOLDER_PATH,
        help="Folder to load models from",
    )

    parser.add_argument(
        "--data-folder",
        type=str,
        required=False,
        default=DATA_FOLDER_PATH,
        help="Data folder",
    )

    parser.add_argument(
        "--load-fields",
        action='store_true',
        required=False,
        help="Load fields instead of creating dataset in ordinary way (read all csv files, build vocab etc.)",
    )

    parser.add_argument(
        "--gpu", required=False, action="store_true", help="whether to run in the GPU"
    )

    return parser.parse_args()


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_model(model_name, models_folder, data_folder, load_fields=True, gpu=False):
    # TODO: pass to config

    global model
    global dataset

    if load_fields:
        dataset = Dataset(data_folder=data_folder)
        dataset.load_fields()
    else:
        dataset = Dataset(data_folder=data_folder, init_from_data=True)

    h_dim = H_DIM
    z_dim = Z_DIM
    c_dim = C_DIM
    p_word_dropout = DROPOUT

    model = RNN_VAE(
        dataset.n_vocab,
        h_dim,
        z_dim,
        c_dim,
        p_word_dropout=p_word_dropout,
        pretrained_embeddings=dataset.get_vocab_vectors(),
        freeze_embeddings=True,
        gpu=gpu,
    )

    if gpu:
        model.load_state_dict(torch.load("{}/{}.bin".format(models_folder, model_name)))
    else:  # TODO: DRY
        model.load_state_dict(
            torch.load(
                "{}/{}.bin".format(models_folder, model_name),
                map_location=lambda storage, loc: storage,
            )
        )


def _test_model(gpu=False):
    log("test model")

    global model
    global dataset

    tests.test_generate_text_of_sentiment(model, dataset)
    tests.test_interpolate(model, dataset, gpu)
    tests.test_encode_text_add_noize_and_decode(model, dataset)


def _main():
    args = _parse_args()

    _set_seed(args.seed)
    _load_model(args.model, args.models_folder, args.data_folder,
                args.load_fields, args.gpu)
    # _test_model(args.gpu)

    app.run(host="0.0.0.0", port=args.port, debug=True, threaded=True)


if __name__ == "__main__":
    _main()
