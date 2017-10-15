import numpy as np

import sys
sys.path.append('..')
from utils import train


def main():

    data = np.load('data.npy')
    unigram_distribution = np.load('unigram_distribution.npy')
    word_vectors = np.load('word_vectors.npy')

    train(
        data, unigram_distribution, word_vectors,
        batch_size=4096, n_topics=30, lambda_const=100.0, num_sampled=15,
        topics_lr=1e-3, doc_weights_lr=1e-3, word_vecs_lr=1e-3, n_epochs=200,
        save_every_n_epochs=10, grad_clip=5.0
    )


main()
