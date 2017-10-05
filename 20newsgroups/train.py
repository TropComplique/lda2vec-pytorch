import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import sys
sys.path.append('..')
from utils import negative_sampling_loss, topic_embedding


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
    
    # create data feeder
    batch_size = 4096
    dataset = SimpleDataset(data)
    iterator = DataLoader(
        dataset, batch_size=batch_size, num_workers=4,
        shuffle=True, pin_memory=True, drop_last=True
    )
    data_size = len(data)
    
    n_topics = 20
    embedding_dim = word_vectors.shape[1]
    vocab_size = len(unigram_distribution)
    window_size = 10
    num_sampled = 15
    
    topics = topic_embedding(n_topics, embedding_dim)
    model = loss(
        topics, word_vectors, unigram_distribution, 
        n_documents, n_topics, num_sampled
    )
    model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 150
    n_batches = math.floor(data_size/batch_size)
    
    model.train()
    try:
        for epoch in range(1, n_epochs + 1):
            print(epoch)
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
                    p.grad = p.grad.clamp(min=-5.0, max=5.0)

                optimizer.step()

            print('{0:.2f} {1:.2f}'.format(
                neg_loss.data[0], dirichlet_loss.data[0]
            ))
            if epoch % 5 == 0:
                print('saving!')
                torch.save(model.state_dict(), 'tmp_model_state.pytorch')
    
    except (KeyboardInterrupt, SystemExit):
        print(' Interruption detected, exiting the program...')
    
    torch.save(model.state_dict(), 'model_state.pytorch')        


class loss(nn.Module):
    """The main thing to be minimized"""

    def __init__(self, topics, word_vectors, unigram_distribution, 
                 n_documents, n_topics, num_sampled):
        super(loss, self).__init__()
        
        # document distributions over the topics 
        self.doc_weights = nn.Embedding(n_documents, n_topics)
        self.doc_weights.weight = nn.Parameter(2.0*torch.rand(n_documents, n_topics) - 1.0)
        
        self.neg = negative_sampling_loss(word_vectors, unigram_distribution, num_sampled)
        self.topics = topics
        self.n_topics = n_topics

    def forward(self, doc_indices, pivot_words, target_words):
        
        alpha = 1.0/self.n_topics
        lambda_const = 1.0
        
        # shape: [batch_size, n_topics]
        doc_weights = self.doc_weights(doc_indices)
        
        # shape: [batch_size, embedding_dim]
        doc_vectors = self.topics(doc_weights)
        
        neg_loss = self.neg(pivot_words, target_words, doc_vectors)
        dirichlet_loss = lambda_const*(1.0 - alpha)*F.log_softmax(doc_weights).sum(1).mean()

        return neg_loss, dirichlet_loss


class SimpleDataset(Dataset):

    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

main()
