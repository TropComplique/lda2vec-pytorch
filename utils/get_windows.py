

def get_windows(doc, hws=5):
    """
    For each word in a document get a window around it.

    Arguments:
        doc: a list of words.
        hws: an integer, half window size.

    Returns:
        a list of tuples, each tuple looks like this
            (word w, window around w),
            window around w equals to
            [hws words that come before w] + [hws words that come after w],
            size of the window around w is 2*hws.
            Number of the tuples = len(doc).
    """
    length = len(doc)
    assert length > 2*hws, 'doc is too short!'

    inside = [(w, doc[(i - hws):i] + doc[(i + 1):(i + hws + 1)])
              for i, w in enumerate(doc[hws:-hws], hws)]

    # for words that are near the beginning or
    # the end of a doc tuples are slightly different
    beginning = [(w, doc[:i] + doc[(i + 1):(2*hws + 1)])
                 for i, w in enumerate(doc[:hws], 0)]

    end = [(w, doc[-(2*hws + 1):i] + doc[(i + 1):])
           for i, w in enumerate(doc[-hws:], length - hws)]

    return beginning + inside + end
