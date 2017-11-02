import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from scipy.stats import ortho_group
from .alias_multinomial import AliasMultinomial


# a small value
EPSILON = 1e-9

PIVOTS_DROPOUT = 0.5
DOC_VECS_DROPOUT = 0.25
DOC_WEIGHTS_INIT = 0.1


class loss(nn.Module):
    """The main thing to minimize."""

    def __init__(self, topics, word_vectors, unigram_distribution,
                 n_documents, loss_doc_weights, lambda_const=100.0, num_sampled=15):
        """
        Arguments:
            topics: An instance of 'topic_embedding' class.
            word_vectors: A float tensor of shape [vocab_size, embedding_dim].
                A word embedding.
            unigram_distribution: A float tensor of shape [vocab_size]. A distribution
                from which to sample negative words.
            n_documents: An integer, number of documents in dataset.
            loss_doc_weights: A float tensor with shape [n_documents],
                for weighting each document when computing loss
                before taking average over a batch.
            lambda_const: A float number, strength of dirichlet prior.
            num_sampled: An integer, number of negative words to sample.
        """
        super(loss, self).__init__()

        self.topics = topics
        self.n_topics = topics.n_topics
        self.alpha = 1.0/self.n_topics
        self.lambda_const = lambda_const
        self.weights = loss_doc_weights

        # document distributions (logits) over the topics
        self.doc_weights = nn.Embedding(n_documents, self.n_topics)
        init.normal(self.doc_weights.weight, std=DOC_WEIGHTS_INIT)

        self.neg = negative_sampling_loss(word_vectors, unigram_distribution, num_sampled)

    def forward(self, doc_indices, pivot_words, target_words):
        """
        Arguments:
            doc_indices: A long tensor of shape [batch_size].
            pivot_words: A long tensor of shape [batch_size].
            target_words: A long tensor of shape [batch_size, window_size].
        Returns:
            A pair of losses, their sum is going to be minimized.
        """

        # shape: [batch_size, n_topics]
        doc_weights = self.doc_weights(doc_indices)

        # for reweighting loss
        w = Variable(self.weights[doc_indices.data])
        w /= w.sum()
        w *= w.size(0)

        # shape: [batch_size, embedding_dim]
        doc_vectors = self.topics(doc_weights)

        neg_loss = self.neg(pivot_words, target_words, doc_vectors, w)
        dirichlet_loss = (w*F.log_softmax(doc_weights).sum(1)).mean()
        dirichlet_loss *= self.lambda_const*(1.0 - self.alpha)

        return neg_loss, dirichlet_loss


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
        self.embedding.weight.data = word_vectors

        # 'AliasMultinomial' is a lot faster than torch.multinomial
        self.multinomial = AliasMultinomial(word_distribution)

        self.num_sampled = num_sampled
        self.embedding_dim = embedding_dim
        self.dropout1 = nn.Dropout(PIVOTS_DROPOUT)
        self.dropout2 = nn.Dropout(DOC_VECS_DROPOUT)

    def forward(self, pivot_words, target_words, doc_vectors, loss_doc_weights):
        """
        Arguments:
            pivot_words: A long tensor of shape [batch_size].
            target_words: A long tensor of shape [batch_size, window_size].
                Windows around pivot words.
            doc_vectors: A float tensor of shape [batch_size, embedding_dim].
                Documents embeddings.
            loss_doc_weights: A float tensor of shape [batch_size].

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
        noise = self.multinomial.draw(batch_size*window_size*self.num_sampled)
        noise = Variable(noise).view(batch_size, window_size*self.num_sampled)

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
        return (loss_doc_weights*neg_loss.sum(1)).mean().neg()


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
        self.n_topics = n_topics

    def forward(self, doc_weights):
        """Embed a batch of documents.

        Arguments:
            doc_weights: A float tensor of shape [batch_size, n_topics],
                document distributions (logits) over the topics.

        Returns:
            A float tensor of shape [batch_size, embedding_dim].
        """

        doc_probs = F.softmax(doc_weights)

        # shape: [batch_size, n_topics, 1]
        unsqueezed_doc_probs = doc_probs.unsqueeze(2)

        # shape: [1, n_topics, embedding_dim]
        unsqueezed_topic_vectors = self.topic_vectors.unsqueeze(0)

        # linear combination of topic vectors weighted by probabilities,
        # shape: [batch_size, embedding_dim]
        doc_vectors = (unsqueezed_doc_probs*unsqueezed_topic_vectors).sum(1)

        return doc_vectors
