import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import math
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from .lda2vec_loss import loss, topic_embedding


BETA = 3.0/4.0


def train(
        data, unigram_distribution, word_vectors,
        batch_size=4096, n_topics=30, lambda_const=100.0, num_sampled=15,
        topics_lr=1e-3, doc_weights_lr=1e-3, word_vecs_lr=1e-3, n_epochs=200,
        save_every_n_epochs=10, grad_clip=5.0):
    """Trains a lda2vec model. Saves the trained model and logs.

    'data' consists of windows around words. Each row in 'data' contains:
    id of a document, id of a word, 'window_size' words around the word.

    Arguments:
        data: A numpy int array with shape [n_windows, window_size + 2].
        unigram_distribution: A numpy float array with shape [vocab_size].
        word_vectors: A numpy float array with shape [vocab_size, embedding_dim].
        batch_size: An integer.
        n_topics: An integer.
        lambda_const: A float number, strength of dirichlet prior.
        num_sampled: An integer, number of negative words to sample.
        topics_lr: A float number, learning rate for topic vectors.
        doc_weights_lr: A float number, learning rate for document weights.
        word_vecs_lr: A float number, learning rate for word vectors.
        n_epochs: An integer.
        save_every_n_epochs: An integer, save the model from time to time.
        grad_clip: A float number, clip gradients by absolute value.
    """

    n_windows = len(data)
    n_documents = len(np.unique(data[:, 0]))
    embedding_dim = word_vectors.shape[1]
    vocab_size = len(unigram_distribution)
    print('number of documents:', n_documents)
    print('number of windows:', n_windows)
    print('number of topics:', n_topics)
    print('vocabulary size:', vocab_size)
    print('word embedding dim:', embedding_dim)

    # prepare word distribution
    unigram_distribution = torch.FloatTensor(unigram_distribution**BETA)
    unigram_distribution /= unigram_distribution.sum()
    unigram_distribution = unigram_distribution.cuda()

    # create a data feeder
    dataset = SimpleDataset(torch.LongTensor(data))
    iterator = DataLoader(
        dataset, batch_size=batch_size, num_workers=4,
        shuffle=True, pin_memory=True, drop_last=True
    )

    # create a lda2vec model
    topics = topic_embedding(n_topics, embedding_dim)
    word_vectors = torch.FloatTensor(word_vectors)
    model = loss(
        topics, word_vectors, unigram_distribution,
        n_documents, lambda_const, num_sampled
    )
    model.cuda()

    # temperature = 5.0
    # doc_weights_init = np.load('doc_weights_init.npy')
    # doc_weights_init += np.random.uniform(high=0.1, size=doc_weights_init.shape)
    # doc_weights_init = np.log(doc_weights_init/doc_weights_init.sum(1, keepdims=True))
    # doc_weights_init /= temperature
    # model.doc_weights.weight.data = torch.FloatTensor(doc_weights_init).cuda()

    params = [
        {'params': model.topics.topic_vectors, 'lr': topics_lr, 'weight_decay': 1e-3},
        {'params': model.doc_weights.weight, 'lr': doc_weights_lr},
        {'params': model.neg.embedding.weight, 'lr': word_vecs_lr}
    ]
    optimizer = optim.Adam(params)
    n_batches = math.floor(n_windows/batch_size)
    print('number of batches:', n_batches, '\n')
    losses = []  # collect all losses here

    model.train()
    try:
        for epoch in range(1, n_epochs + 1):
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
                    p.grad = p.grad.clamp(min=-grad_clip, max=grad_clip)

                optimizer.step()

            print('{0:.2f} {1:.2f}'.format(
                neg_loss.data[0], dirichlet_loss.data[0]
            ))
            losses += [(epoch, neg_loss.data[0], dirichlet_loss.data[0])]
            if epoch % save_every_n_epochs == 0:
                print('\nsaving!\n')
                torch.save(model.state_dict(), str(epoch) + '_epoch_model_state.pytorch')

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


def doc_by_doc_iterator(data):
    docs = []
    unique_docs = np.unique(data[:, 0])
    for i in unique_docs:
        doc = data[data[:, 0] == i]
        docs.append(torch.FloatTensor(doc))
    return docs
