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
LAMBDA_CONST = 200.0 # 20
WORD_VECS_LR = 1e-3
DOC_WEIGHTS_LR = 1e-3
TOPICS_LR = 1e-3
NUM_SAMPLED = 15
N_TOPICS = 20
N_EPOCHS = 200
GRAD_CLIP = 1.0


def main():

    # load data
    data = np.load('data.npy')
    unigram_distribution = np.load('unigram_distribution.npy')[()]
    word_vectors = np.load('word_vectors.npy')
    n_documents = len(np.unique(data[:, 0]))
    data_size = len(data)

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

    # set parameters
    embedding_dim = word_vectors.shape[1]
    vocab_size = len(unigram_distribution)
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

    # temperature = 5.0
    # doc_weights_init = np.load('doc_weights_init.npy')
    # doc_weights_init += np.random.uniform(high=0.1, size=doc_weights_init.shape)
    # doc_weights_init = np.log(doc_weights_init/doc_weights_init.sum(1, keepdims=True))
    # doc_weights_init /= temperature
    # model.doc_weights.weight.data = torch.FloatTensor(doc_weights_init).cuda()

    params = [
        {'params': model.topics.topic_vectors, 'lr': TOPICS_LR, 'weight_decay': 1e-3},
        {'params': model.doc_weights.weight, 'lr': DOC_WEIGHTS_LR},
        {'params': model.neg.embedding.weight, 'lr': WORD_VECS_LR}
    ]
    optimizer = optim.Adam(params)
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
            if epoch % 10 == 0:
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
