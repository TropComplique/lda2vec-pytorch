# lda2vec
Pytorch implementation of Moody's lda2vec, a way of topic modeling using word embeddings. The original paper:
[Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec](https://arxiv.org/abs/1605.02019).

**Warning:**
I, personally, believe that lda2vec algorithm isn't working as presented in the original paper.

## Loss
The following objective function is maximized

![objective function](loss.png)

where `c` - context vector, `w` - word vector, `lambda` - positive constant that controls sparsity, `i` - sum over the window around the word, `k` - sum over sampled negative words, `j` - sum over the topics, `p` - probability distribution over the topics, `t` - topic vectors.

## Implementation details
* I use vanilla LDA to initialize lda2vec. It is not like in the original paper.
* I add noise to gradients.
* I reweight loss according to document lengths.
* I train 50 dimensional skip-gram word2vec before training lda2vec.
* For text preprocessing I lemmatize and then remove rare and frequent words.

## Requirements
* pytorch 0.2, spacy 1.9, gensim 3.0
* numpy, sklearn, tqdm
* matplotlib, [Multicore-TSNE](https://github.com/DmitryUlyanov/Multicore-TSNE)
