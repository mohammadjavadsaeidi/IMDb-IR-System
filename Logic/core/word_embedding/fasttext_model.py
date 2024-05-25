import fasttext
import re
import string

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
import numpy as np

from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.preprocessing import preprocess_text

class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None

    def train(self, texts):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        with open('train_data.txt', 'w') as f:
            for text in texts:
                f.write(preprocess_text(text) + '\n')

        self.model = fasttext.train_unsupervised('train_data.txt', model=self.method)

    def get_query_embedding(self, query, do_preprocess=True):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        if do_preprocess:
            query = preprocess_text(query)
        return self.model.get_sentence_vector(query)

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        vec = self.model.get_word_vector(word2) - self.model.get_word_vector(word1) + self.model.get_word_vector(word3)
        words = self.model.get_words()
        min_dist = float('inf')
        best_word = None
        for word in words:
            if word not in [word1, word2, word3]:
                dist = distance.cosine(vec, self.model.get_word_vector(word))
                if dist < min_dist:
                    min_dist = dist
                    best_word = word
        return best_word

    def save_model(self, path='FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)

    def load_model(self, path="FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, save=False, path='FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)


if __name__ == "__main__":
    ft_model = FastText(method='skipgram')

    path = '/Users/snapp/PycharmProjects/IMDb-IR-System/Logic/core/indexer/index.json/documents.json'
    ft_data_loader = FastTextDataLoader(path)

    X, y = ft_data_loader.create_train_data()

    ft_model.train(X)
    ft_model.prepare(None, mode="save")

    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "king"
    word3 = "queen"
    print(
        f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
