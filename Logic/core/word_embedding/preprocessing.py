import string
import time

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                    punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """

    if lower_case:
        text = text.lower()

    if punctuation_removal:
        text = text.translate(str.maketrans('', '', string.punctuation))

    tokens = word_tokenize(text)

    if stopword_removal:
        stop_words = set(stopwords.words('english') + stopwords_domain)
        tokens = [word for word in tokens if word not in stop_words]

    tokens = [word for word in tokens if len(word) >= minimum_length]

    return ' '.join(tokens)