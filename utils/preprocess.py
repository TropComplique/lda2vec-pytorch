from collections import Counter
from tqdm import tqdm


def preprocess(docs, nlp, min_length=11, min_counts=5):
    """
    Arguments:
        docs: a list of tuples (index, string), each string is a document.
        nlp: a spaCy object, like nlp = spacy.load('en').
        min_length: an integer, minimum document length.
        min_counts: an integer.

    Returns:
        encoded_docs: a list of tuples (index, list), each list is a document encoded
            by integer values.
        decoder: a dict, integer -> word.
        word_counts: a list of integers, counts of words that are in decoder.
            word_counts[i] is the number of occurrences of word decoder[i]
            in all documents in docs.
    """

    def clean_and_tokenize(doc):
        text = ' '.join(doc.split())  # remove excessive spaces
        text = nlp(text, tag=True, parse=False, entity=False)
        return [t.lemma_ for t in text
                if t.is_alpha and len(t) > 2 and (not t.is_stop)]

    tokenized_docs = [(i, clean_and_tokenize(doc)) for i, doc in tqdm(docs)]

    # remove short documents
    n_short_docs = sum(1 for i, doc in tokenized_docs if len(doc) < min_length)
    tokenized_docs = [(i, doc) for i, doc in tokenized_docs if len(doc) >= min_length]
    print('number of removed short documents:', n_short_docs)

    counts = _count_unique_tokens(tokenized_docs)
    tokenized_docs = _remove_rare_tokens(counts, min_counts, tokenized_docs)
    n_short_docs = sum(1 for i, doc in tokenized_docs if len(doc) < min_length)
    tokenized_docs = [(i, doc) for i, doc in tokenized_docs if len(doc) >= min_length]
    print('number of additionally removed short documents:', n_short_docs)

    counts = _count_unique_tokens(tokenized_docs)
    encoder, decoder, word_counts = _create_token_encoder(counts)
    
    print('\nminimum word count number:', word_counts[-1])
    print('this number can be less than MIN_COUNTS because of document removal')
    
    encoded_docs = _encode(tokenized_docs, encoder)
    return encoded_docs, decoder, word_counts


def _count_unique_tokens(tokenized_docs):
    tokens = []
    for i, doc in tokenized_docs:
        tokens += doc
    return Counter(tokens)


def _encode(tokenized_docs, encoder):
    result = []
    for i, doc in tokenized_docs:
        result.append((i, [encoder[t] for t in doc]))
    return result


def _remove_rare_tokens(counts, min_counts, tokenized_docs):

    # words with count < min_counts
    # will be removed

    total_tokens_count = sum(
        count for token, count in counts.most_common()
    )
    print('total number of tokens:', total_tokens_count)

    # number of tokens that will be removed
    unknown_tokens_count = sum(
        count for token, count in counts.most_common()
        if count < min_counts
    )
    print('number of unknown tokens to be removed:', unknown_tokens_count)

    keep = {}
    for token, count in counts.most_common():
        keep[token] = count >= min_counts

    return [(i, [t for t in doc if keep[t]]) for i, doc in tokenized_docs]


def _create_token_encoder(counts):
    
    total_tokens_count = sum(
        count for token, count in counts.most_common()
    )
    print('total number of tokens:', total_tokens_count)

    encoder = {}
    decoder = {}
    word_counts = []
    i = 0

    for token, count in counts.most_common():
        # counts.most_common() is in decreasing count order
        encoder[token] = i
        decoder[i] = token
        word_counts.append(count)
        i += 1

    return encoder, decoder, word_counts
