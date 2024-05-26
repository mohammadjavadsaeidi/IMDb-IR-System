import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
from scipy.spatial import distance

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.x_train = x
        self.y_train = y
        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        predictions = []
        for instance in tqdm(x):
            distances = np.array([distance.euclidean(instance, train_instance) for train_instance in self.x_train])
            nearest_neighbors_indices = distances.argsort()[:self.k]
            nearest_neighbors_classes = [self.y_train[idx] for idx in nearest_neighbors_indices]
            unique, counts = np.unique(nearest_neighbors_classes, return_counts=True)
            predicted_class = unique[counts.argmax()]
            predictions.append(predicted_class)
        return np.array(predictions)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        predictions = self.predict(x)
        return classification_report(y, predictions)


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """

    file_path = './IMDB_Dataset.csv'
    fasttext_model_path = '/Users/snapp/PycharmProjects/IMDb-IR-System/Logic/core/word_embedding/FastText_model.bin'
    review_loader = ReviewLoader(file_path, fasttext_model_path)
    review_loader.load_data()
    review_loader.get_embeddings()
    x_train, x_test, y_train, y_test = review_loader.split_data()

    classifier = KnnClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)

    report = classifier.prediction_report(x_test, y_test)
    print(report)
