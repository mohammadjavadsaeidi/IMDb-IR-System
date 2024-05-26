import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Logic.core.word_embedding.fasttext_model import FastText
from Logic.core.word_embedding.preprocessing import preprocess_text


class ReviewLoader:
    def __init__(self, file_path: str, fasttext_model_path: str):
        self.file_path = file_path
        self.fasttext_model_path = fasttext_model_path
        self.fasttext_model = FastText()
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        # Load the data
        data = pd.read_csv(self.file_path)

        # Preprocess reviews
        self.review_tokens = data['review'].apply(preprocess_text).tolist()

        # Encode sentiments
        le = LabelEncoder()
        self.sentiments = le.fit_transform(data['sentiment'])

        # Load FastText model
        if os.path.exists(self.fasttext_model_path):
            self.fasttext_model.load_model(self.fasttext_model_path)
        else:
            raise FileNotFoundError(f"FastText model file '{self.fasttext_model_path}' not found.")

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        self.embeddings = [self.fasttext_model.get_query_embedding(review) for review in tqdm(self.review_tokens)]

    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        x_train, x_test, y_train, y_test = train_test_split(self.embeddings, self.sentiments, test_size=test_data_ratio,
                                                            random_state=42)
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


# Example usage
if __name__ == '__main__':
    file_path = './IMDB_Dataset.csv'
    fasttext_model_path = '/Users/snapp/PycharmProjects/IMDb-IR-System/Logic/core/word_embedding/FastText_model.bin'
    review_loader = ReviewLoader(file_path, fasttext_model_path)
    review_loader.load_data()
    review_loader.get_embeddings()
    x_train, x_test, y_train, y_test = review_loader.split_data()

    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)
