import string
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        self.documents = documents
        self.stopwords = load_stopwords()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        preprocessed_documents = []
        for doc in self.documents:
            doc = self.remove_stopwords(doc)
            doc = self.remove_links(doc)
            doc = self.normalize(doc)
            doc = self.remove_punctuations(doc)
            preprocessed_documents.append(doc)

        return preprocessed_documents

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """

        text = text.lower()
        text = ' '.join([self.lemmatizer.lemmatize(work.lower()) for work in self.tokenize(text)])
        text = ' '.join([self.stemmer.stem(word) for word in self.tokenize(text)])
        return text

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        translator = str.maketrans(dict.fromkeys(list(string.punctuation)))
        return text.translate(translator)

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        return word_tokenize(text)

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        return ' '.join([word for word in self.tokenize(text) if word.lower() not in self.stopwords])


def load_stopwords():
    """
        Load stopwords from a file.

        Returns
        ----------
        set
            A set of stopwords.
        """
    with open("./stopwords.txt", 'r') as file:
        return file.read().splitlines()


if __name__ == '__main__':
    print(Preprocessor(['spider man is wonderful where google.com regression']).preprocess())