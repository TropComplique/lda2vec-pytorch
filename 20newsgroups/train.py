import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import math
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import sys
sys.path.append('..')
from utils import loss, topic_embedding


BATCH_SIZE = 4096
LAMBDA_CONST = 200.0
LEARNING_RATE = 1e-3
TOPICS_LR = 1e-4
NUM_SAMPLED = 15
N_TOPICS = 20
N_EPOCHS = 250
GRAD_CLIP = 5.0


def main():

    # load data
    data = np.load('data.npy')
    unigram_distribution = np.load('unigram_distribution.npy')[()]
    word_vectors = np.load('word_vectors.npy')
    n_documents = len(np.unique(data[:, 0]))

    # convert to pytorch tensors
    data = torch.LongTensor(data)
    word_vectors = torch.FloatTensor(word_vectors)
    unigram_distribution = torch.FloatTensor(unigram_distribution**(3.0/4.0))
    unigram_distribution /= unigram_distribution.sum()
    unigram_distribution = unigram_distribution.cuda()

    # create a data feeder
    dataset = SimpleDataset(data)
    iterator = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=4,
        shuffle=True, pin_memory=True, drop_last=True
    )
    data_size = len(data)

    # set parameters
    embedding_dim = word_vectors.shape[1]
    vocab_size = len(unigram_distribution)
    window_size = 10  # fixed parameter, it depends on the data
    print('number of documents:', n_documents)
    print('number of windows:', data_size)
    print('number of topics:', N_TOPICS)
    print('vocabulary size:', vocab_size)
    print('word embedding dim:', embedding_dim)

    # create a lda2vec model
    topics = topic_embedding(N_TOPICS, embedding_dim)
    model = loss(
        topics, word_vectors, unigram_distribution,
        n_documents, LAMBDA_CONST, NUM_SAMPLED
    )
    model.cuda()

    params = [
        {'params': model.topics.topic_vectors, 'lr': TOPICS_LR},
        {'params': model.doc_weights.weight},
        {'params': model.neg.embedding.weight}
    ]
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    n_batches = math.floor(data_size/BATCH_SIZE)
    print('number of batches:', n_batches, '\n')
    losses = []  # collect all losses here

    model.train()
    try:
        for epoch in range(1, N_EPOCHS + 1):
            print('epoch', epoch)
            for batch in tqdm(iterator):

                batch = Variable(batch.cuda())
                doc_indices = batch[:, 0]
                pivot_words = batch[:, 1]
                target_words = batch[:, 2:]

                neg_loss, dirichlet_loss = model(doc_indices, pivot_words, target_words)
                total_loss = neg_loss + dirichlet_loss

                optimizer.zero_grad()
                total_loss.backward()

                # gradient clipping
                for p in model.parameters():
                    p.grad = p.grad.clamp(min=-GRAD_CLIP, max=GRAD_CLIP)

                optimizer.step()

            print('{0:.2f} {1:.2f}'.format(
                neg_loss.data[0], dirichlet_loss.data[0]
            ))
            losses += [(epoch, neg_loss.data[0], dirichlet_loss.data[0])]
            if epoch % 20 == 0:
                print('\nsaving!\n')
                torch.save(model.state_dict(), str(epoch) + '_tmp_model_state.pytorch')

    except (KeyboardInterrupt, SystemExit):
        print(' Interruption detected, exiting the program...')

    _write_training_logs(losses)
    torch.save(model.state_dict(), 'model_state.pytorch')


class SimpleDataset(Dataset):

    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def _write_training_logs(losses):
    with open('training_logs.txt', 'w') as f:
        column_names = 'epoch,negative_sampling_loss,dirichlet_prior_loss\n'
        f.write(column_names)

        for i in losses:
            values = ('{0},{1:.3f},{2:.3f}\n').format(*i)
            f.write(values)


main()
