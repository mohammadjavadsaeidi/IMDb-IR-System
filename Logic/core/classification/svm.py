import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from Logic.core.classification.data_loader import ReviewLoader
from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC()

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

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
        return self.model.predict(x)

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


# F1 accuracy : 78%
if __name__ == '__main__':
    file_path = './IMDB_Dataset.csv'
    fasttext_model_path = '/Users/snapp/PycharmProjects/IMDb-IR-System/Logic/core/word_embedding/FastText_model.bin'
    review_loader = ReviewLoader(file_path, fasttext_model_path)
    review_loader.load_data()
    review_loader.get_embeddings()
    x_train, x_test, y_train, y_test = review_loader.split_data()

    classifier = SVMClassifier()
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)

    report = classification_report(y_test, predictions)
    print(report)
