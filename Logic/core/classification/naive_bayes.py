import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from Logic.core.classification.basic_classifier import BasicClassifier


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.number_of_samples, self.number_of_features = x.shape

        self.prior = np.zeros(self.num_classes)
        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))

        for i, cls in enumerate(self.classes):
            x_class = x[y == cls]
            self.prior[i] = x_class.shape[0] / self.number_of_samples
            self.feature_probabilities[i, :] = (np.sum(x_class, axis=0) + self.alpha) / (
                        x_class.shape[0] + 2 * self.alpha)

        self.log_probs = np.log(self.feature_probabilities)

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
        log_prior = np.log(self.prior)
        log_likelihood = x @ self.log_probs.T
        log_posterior = log_likelihood + log_prior
        return self.classes[np.argmax(log_posterior, axis=1)]

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

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        x = self.cv.transform(sentences)
        predictions = self.predict(x)
        positive_count = sum(predictions)
        total_count = len(predictions)
        return (positive_count / total_count) * 100 if total_count > 0 else 0.0


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """

    data_path = './IMDB_Dataset.csv'
    data = pd.read_csv(data_path)

    x = data['review'].tolist()
    y = data['sentiment'].apply(lambda sentiment: 1 if sentiment == 'positive' else 0).tolist()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    cv = CountVectorizer()
    x_train_cv = cv.fit_transform(x_train).toarray()
    x_test_cv = cv.transform(x_test).toarray()

    classifier = NaiveBayes(count_vectorizer=cv)
    classifier.fit(x_train_cv, y_train)

    report = classifier.prediction_report(x_test_cv, y_test)
    print(report)
