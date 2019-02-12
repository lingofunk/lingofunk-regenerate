import torch
import numpy as np
from lingofunk_regenerate.utils import log as _log


def log(text):
    _log(text, prefix='tests: ')


def test_generate_text_of_sentiment(model, dataset):
    log('Text generation of given sentiment\n')

    # Samples latent and conditional codes randomly from prior
    z = model.sample_z_prior(1)
    c = model.sample_c_prior(1)

    # Generate positive sample given z
    c[0, 0], c[0, 1] = 1, 0

    _, c_idx = torch.max(c, dim=1)
    sample_idxs = model.sample_sentence(z, c, temp=0.1)

    log('Sentiment: {}'.format(dataset.idx2label(int(c_idx))))
    log('Generated: {}\n'.format(dataset.idxs2sentence(sample_idxs)))

    # Generate negative sample from the same z
    c[0, 0], c[0, 1] = 0, 1

    _, c_idx = torch.max(c, dim=1)
    sample_idxs = model.sample_sentence(z, c, temp=0.8)

    log('Sentiment: {}'.format(dataset.idx2label(int(c_idx))))
    log('Generated: {}\n'.format(dataset.idxs2sentence(sample_idxs)))


def test_interpolate(model, dataset, gpu=False):
    log('Interpolation of z\n')

    c = model.sample_c_prior(1)

    z1 = model.sample_z_prior(1).view(1, 1, model.z_dim)
    z1 = z1.cuda() if gpu else z1

    z2 = model.sample_z_prior(1).view(1, 1, model.z_dim)
    z2 = z2.cuda() if gpu else z2

    # Interpolation coefficients
    alphas = np.linspace(0, 1, 10)

    for alpha in alphas:
        z = float(1-alpha) * z1 + float(alpha) * z2

        sample_idxs = model.sample_sentence(z, c, temp=0.1)
        sample_sent = dataset.idxs2sentence(sample_idxs)

        log("{}".format(sample_sent))


def test_encode_text_add_noize_and_decode(model, dataset, sentence='I love to eat in Las Vegas', use_c_prior=True):
    log('Encode, add noize & decode\n')
    log('Sentence: ' + sentence)

    mbsize = 1

    sentence = dataset.sentence2idxs(sentence)
    sentence = np.array(sentence).reshape(-1, mbsize)
    sentence = torch.from_numpy(sentence)

    # Encoder: sentence -> z
    mu, logvar = model.forward_encoder(sentence)

    if use_c_prior:
        c = model.sample_c_prior(mbsize)
    else:
        c = model.forward_discriminator(sentence.transpose(0, 1))
    # c[0, 0], c[0, 1] = 1, 0

    sigma = 0.001
    temp = 0.001

    for i in range(10):
        z = model.sample_z(mu, logvar)

        z_noized = z.type(torch.FloatTensor) + \
             torch.from_numpy(np.random.randn(z.size(0), z.size(1))).type(torch.FloatTensor) * (0 if i == 0 else sigma)

        sampled = model.sample_sentence(z_noized, c, temp=temp)
        sampled_sentence = dataset.idxs2sentence(sampled)
        print('Sampled sentence: ' + sampled_sentence + '\n')
