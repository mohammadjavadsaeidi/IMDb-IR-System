import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from Logic.core.word_embedding.preprocessing import preprocess_text


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """

    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        data = pd.read_json(self.file_path)

        # Extract movie details
        records = []
        for movie_id, movie_details in data.items():
            reviews = movie_details.get('reviews', [])
            # Convert list of lists to list of strings
            if reviews is None:
                continue
            review_texts = [' '.join(review) for review in reviews]
            # Join all review texts into a single string

            review_text = ' '.join(review_texts)

            record = {
                'title': movie_details.get('title', ''),
                'genres': ''.join(movie_details.get('genres', '')),
                'synopsis': movie_details.get('synopsis', ''),
                'summaries': movie_details.get('summaries', ''),
                'reviews': review_text
            }
            records.append(record)

        return pd.DataFrame(records)

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()

        # Preprocess text data
        df['text'] = str(df['synopsis'].fillna('')) + ' ' + str(df['summaries'].fillna('')) + ' ' + str(
            df['reviews'].fillna(
                '')) + ' ' + df['title'].fillna('')
        df['text'] = df['text'].apply(lambda x: preprocess_text(x))

        # Encode labels
        le = LabelEncoder()
        df['genres'] = le.fit_transform(df['genres'])

        X = df['text'].values
        y = df['genres'].values

        return X, y
