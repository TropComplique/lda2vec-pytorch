from collections import Counter
from tqdm import tqdm
import re


def preprocess(docs, nlp, min_length=11, min_counts=5):
    """
    Note: all rare words will be replaced by token <UNK>.

    Arguments:
        docs: a list of strings, each string is a document.
        nlp: a spaCy object, like nlp = spacy.load('en').
        min_length: an integer.
        min_counts: an integer.

    Returns:
        encoded_docs: a list of lists, each list is a document encoded
            by integer values.
        decoder: a dict, integer -> word.
        word_counts: a list of integers, counts of words that are in decoder.
            word_counts[i] is the number of occurrences of word decoder[i]
            in all documents in docs.
    """

    spaces = re.compile(r' +')
    nonletters = re.compile(r'[^a-z A-Z]+')
    shortwords = re.compile(r'\b\w{1,3}\b')

    def clean_and_tokenize(doc):
        text = re.sub(nonletters, ' ', doc)
        text = re.sub(shortwords, ' ', text)
        text = re.sub(spaces, ' ', text).strip()
        text = nlp(text, tag=True, parse=False, entity=False)
        return [t.lemma_ for t in text if not t.is_stop]

    tokenized_docs = [clean_and_tokenize(doc) for doc in tqdm(docs)]

    # remove short documents
    tokenized_docs = [doc for doc in tokenized_docs if len(doc) >= min_length]

    counts = count_unique_tokens(tokenized_docs)
    encoder, decoder, word_counts = create_token_encoder(counts, min_counts)
    encoded_docs = encode(tokenized_docs, encoder)
    return encoded_docs, decoder, word_counts


def count_unique_tokens(tokenized_docs):
    tokens = []
    for doc in tokenized_docs:
        tokens += doc
    return Counter(tokens)


def encode(tokenized_docs, encoder):
    result = []
    for doc in tokenized_docs:
        result.append([encoder[t] for t in doc])
    return result


def create_token_encoder(counts, min_counts):

    # words with count < min_counts will be
    # replaced by <UNK> - unknown token

    # number of tokens that will be replaced
    unknown_tokens_count = sum(
        count for token, count in counts.most_common()
        if count < min_counts
    )
    counts['<UNK>'] = unknown_tokens_count

    # for all words
    encoder = {}

    # only for words with count >= min_counts
    decoder = {}
    word_counts = []

    # encoder is such that words with count >= min_counts will
    # be replaced by unique integers, but words with count < min_counts
    # will be all replaced by one integer (given by encoder['<UNK>'])

    for i, (token, count) in enumerate(counts.most_common()):
        if count >= min_counts:
            encoder[token] = i
            decoder[i] = token
            word_counts.append(count)
        else:
            # this will work if there is
            # more than one word with count < min_counts,
            # it almost always (99.9%) happens
            encoder[token] = encoder['<UNK>']

    assert len(counts) == len(encoder)
    return encoder, decoder, word_counts
