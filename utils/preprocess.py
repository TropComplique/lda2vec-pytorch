from collections import Counter
from tqdm import tqdm


def preprocess(docs, nlp, has_vector, min_length=11, min_counts=5):
    """
    Note: all rare words will be replaced by token <UNK>.

    Arguments:
        docs: a list of strings, each string is a document.
        nlp: a spaCy object, like nlp = spacy.load('en').
        has_vector: a boolean function, returns whether 
            a word has a pretrained word vector.
        min_length: an integer, minimum document length.
        min_counts: an integer.

    Returns:
        encoded_docs: a list of lists, each list is a document encoded
            by integer values.
        decoder: a dict, integer -> word.
        word_counts: a list of integers, counts of words that are in decoder.
            word_counts[i] is the number of occurrences of word decoder[i]
            in all documents in docs.
    """

    def clean_and_tokenize(doc):
        text = ' '.join(doc.split()) # remove excessive spaces
        text = nlp(text, tag=True, parse=False, entity=False)
        return [t.lemma_ for t in text 
                if t.is_alpha and len(t) > 2 and (not t.is_stop)]

    tokenized_docs = [clean_and_tokenize(doc) for doc in tqdm(docs)]

    # remove short documents
    tokenized_docs = [doc for doc in tokenized_docs if len(doc) >= min_length]

    counts = _count_unique_tokens(tokenized_docs)
    encoder, decoder, word_counts = _create_token_encoder(counts, has_vector, min_counts)
    encoded_docs = _encode(tokenized_docs, encoder)
    return encoded_docs, decoder, word_counts


def _count_unique_tokens(tokenized_docs):
    tokens = []
    for doc in tokenized_docs:
        tokens += doc
    return Counter(tokens)


def _encode(tokenized_docs, encoder):
    result = []
    for doc in tokenized_docs:
        result.append([encoder[t] for t in doc])
    return result


def _create_token_encoder(counts, has_vector, min_counts):

    # words with count < min_counts and without a pretrained word vector
    # will be replaced by <UNK> - unknown token

    # number of tokens that will be replaced
    unknown_tokens_count = sum(
        count for token, count in counts.most_common()
        if count < min_counts and not has_vector(token)
    )
    counts['<UNK>'] = unknown_tokens_count

    # for all words
    encoder = {}

    # only for words with count >= min_counts or has_vector=True
    decoder = {}
    word_counts = []
    i = 0
    
    for token, count in counts.most_common():
        # counts.most_common() is in decreasing count order
        if count >= min_counts or has_vector(token):
            encoder[token] = i
            decoder[i] = token
            word_counts.append(count)
            i += 1
        else:
            # this will work if there is
            # more than one word with count < min_counts and
            # without a pretrained word vector,
            # it almost always (99.9%) happens
            encoder[token] = encoder['<UNK>']

    return encoder, decoder, word_counts
