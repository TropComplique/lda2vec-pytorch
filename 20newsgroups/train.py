import numpy as np

import sys
sys.path.append('..')
from utils import train


def main():

    data = np.load('data.npy')
    unigram_distribution = np.load('unigram_distribution.npy')
    word_vectors = np.load('word_vectors.npy')
    doc_weights_init = np.load('doc_weights_init.npy')

    # to prevent taking log of zero
    doc_weights_init += 1.0/n_topics

    # transform to logits
    doc_weights_init = np.log(doc_weights_init/doc_weights_init.sum(1, keepdims=True))

    # make distribution softer
    temperature = 5.0
    doc_weights_init /= temperature

    train(
        data, unigram_distribution, word_vectors,
        doc_weights_init, n_topics=25,
        batch_size=4096, n_epochs=200,
        lambda_const=100.0, num_sampled=15,
        topics_weight_decay=1e-2,
        topics_lr=1e-3, doc_weights_lr=1e-3, word_vecs_lr=1e-3,
        save_every=10, grad_clip=5.0
    )


main()
