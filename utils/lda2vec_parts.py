import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ortho_group


# to prevent taking logarithm of zero
EPSILON = 1e-8


class negative_sampling_loss(nn.Module):

    def __init__(self, word_vectors, word_distribution, num_sampled=10):
        """
        Arguments:
            word_vectors: A float tensor of shape [vocab_size, embedding_dim].
                A word representation like, for example, word2vec or GloVe.
            word_distribution: A float tensor of shape [vocab_size]. A distribution
                from which to sample negative words.
            num_sampled: An integer, number of negative words to sample.
        """
        super(negative_sampling_loss, self).__init__()

        vocab_size, embedding_dim = word_vectors.size()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(word_vectors)

        self.embedding_dim = embedding_dim
        self.word_distribution = word_distribution
        self.num_sampled = num_sampled
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, pivot_words, target_words, doc_vectors):
        """Compute loss.

        Arguments:
            pivot_words: A long tensor of shape [batch_size].
            target_words: A long tensor of shape [batch_size, window_size].
                Windows around pivot words.
            doc_vectors: A float tensor of shape [batch_size, embedding_dim].
                Documents embeddings.

        Returns:
            A scalar.
        """

        batch_size, window_size = target_words.size()

        # shape: [batch_size, embedding_dim]
        pivot_vectors = self.embedding(pivot_words)

        # shapes: [batch_size, embedding_dim]
        pivot_vectors = self.dropout1(pivot_vectors)
        doc_vectors = self.dropout2(doc_vectors)
        context_vectors = doc_vectors + pivot_vectors

        # shape: [batch_size, window_size, embedding_dim]
        targets = self.embedding(target_words)

        # shape: [batch_size, 1, embedding_dim]
        unsqueezed_context = context_vectors.unsqueeze(1)

        # compute dot product between a context vector
        # and each word vector in the window,
        # shape: [batch_size, window_size]
        log_targets = (targets*unsqueezed_context).sum(2).sigmoid()\
            .clamp(min=EPSILON).log()

        # sample negative words for each word in the window,
        # shape: [batch_size*window_size*num_sampled]
        noise = torch.multinomial(
            self.word_distribution, batch_size*window_size*self.num_sampled,
            replacement=True
        )
        noise = Variable(noise.cuda())
        noise = noise.view(batch_size, window_size*self.num_sampled)

        # shape: [batch_size, window_size*num_sampled, embedding_dim]
        noise = self.embedding(noise)
        noise = noise.view(batch_size, window_size, self.num_sampled, self.embedding_dim)

        # shape: [batch_size, 1, 1, embedding_dim]
        unsqueezed_context = context_vectors.unsqueeze(1).unsqueeze(1)

        # compute dot product between a context vector
        # and each negative word's vector for each word in the window,
        # then sum over negative words,
        # shape: [batch_size, window_size]
        sum_log_sampled = (noise*unsqueezed_context).sum(3).neg().sigmoid()\
            .clamp(min=EPSILON).log().sum(2)

        neg_loss = log_targets + sum_log_sampled

        # sum over the window, then take mean over the batch
        # shape: []
        return neg_loss.sum(1).mean().neg()


class topic_embedding(nn.Module):

    def __init__(self, n_topics, embedding_dim):
        """
        Arguments:
            embedding_dim: An integer.
            n_topics: An integer.
        """
        super(topic_embedding, self).__init__()

        # initialize topic vectors by a random orthogonal matrix
        assert n_topics < embedding_dim
        topic_vectors = ortho_group.rvs(embedding_dim)
        topic_vectors = topic_vectors[0:n_topics]
        topic_vectors = torch.FloatTensor(topic_vectors)

        self.topic_vectors = nn.Parameter(topic_vectors)
        self.embedding_dim = embedding_dim

    def forward(self, doc_weights):
        """Embed a batch of documents.

        Arguments:
            doc_weights: A float tensor of shape [batch_size, n_topics],
                document distribution (logits) over the topics.

        Returns:
            A float tensor of shape [batch_size, embedding_dim].
        """

        batch_size, n_topics = doc_weights.size()
        doc_probs = F.softmax(doc_weights)

        # shape: [batch_size, n_topics, 1]
        unsqueezed_doc_probs = doc_probs.unsqueeze(2)

        # shape: [1, n_topics, embedding_dim]
        unsqueezed_topic_vectors = self.topic_vectors.unsqueeze(0)

        # linear combination of topic vectors weighted by probabilities,
        # shape: [batch_size, embedding_dim]
        doc_vectors = (unsqueezed_doc_probs*unsqueezed_topic_vectors).sum(1)

        return doc_vectors
