import numpy as np

import sys
sys.path.append('..')
from utils import train


def main():

    data = np.load('data.npy')
    unigram_distribution = np.load('unigram_distribution.npy')
    word_vectors = np.load('word_vectors.npy')
    doc_weights_init = np.load('doc_weights_init.npy')

    # transform to logits
    doc_weights_init = np.log(doc_weights_init + 1e-4)

    # make distribution softer
    temperature = 7.0
    doc_weights_init /= temperature

    # if you want to train the model like in the original paper
    # set doc_weights_init=None

    train(
        data, unigram_distribution, word_vectors,
        doc_weights_init, n_topics=25,
        batch_size=1024*7, n_epochs=123,
        lambda_const=500.0, num_sampled=15,
        topics_weight_decay=1e-2,
        topics_lr=1e-3, doc_weights_lr=1e-3, word_vecs_lr=1e-3,
        save_every=20, grad_clip=5.0
    )


main()
