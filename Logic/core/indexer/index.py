import time
import os
import json
import copy
from Logic.core.preprocess import Preprocessor
from indexes_enum import Indexes


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            current_index[doc['id']] = doc
        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each term's tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        current_index = {}
        for doc in self.preprocessed_documents:
            for star in doc['stars']:
                for term in star.split():
                    if term not in current_index:
                        current_index[term] = {}
                    current_index[term][doc['id']] = doc['stars'].count(star)
        return current_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each term's tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        current_index = {}
        for doc in self.preprocessed_documents:
            for genre in doc['genres']:
                if genre not in current_index:
                    current_index[genre] = {}
                current_index[genre][doc['id']] = doc['genres'].count(genre)
        return current_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each term's tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        for doc in self.preprocessed_documents:
            for summary in doc['summaries']:
                for term in summary.split():
                    if term not in current_index:
                        current_index[term] = {}
                    current_index[term][doc['id']] = summary.count(term)
        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        Get posting list of a word.

        Parameters
        ----------
        word: str
            Word we want to check.
        index_type: str
            Type of index we want to check (documents, stars, genres, summaries).

        Returns
        ----------
        list
            Posting list of the word (you should return the list of document IDs that contain the word and ignore the tf).
        """
        try:
            return list(self.index[index_type][word].keys())
        except KeyError:
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes.

        Parameters
        ----------
        document : dict
            Document to add to all the indexes.
        """
        self.preprocessed_documents.append(document)
        # Rebuild index
        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes.

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes.
        """
        self.preprocessed_documents = [doc for doc in self.preprocessed_documents if doc['id'] != document_id]
        # Rebuild index
        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def store_index(self, path: str, index_type: str = None):
        """
        Stores the index in a file (such as a JSON file).

        Parameters
        ----------
        path : str
            Path to store the file.
        index_type: str or None
            Type of index we want to store (documents, stars, genres, summaries).
            If None store tiered index.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        if index_type is None:
            with open(os.path.join(path, 'index.json'), 'w') as f:
                json.dump(self.index, f, indent=4)
        elif index_type in self.index:
            with open(os.path.join(path, f'{index_type}.json'), 'w') as f:
                json.dump(self.index[index_type], f, indent=4)
        else:
            raise ValueError('Invalid index type')

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """
        with open(path, 'r') as f:
            self.index = json.load(f)

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(
                set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(
                set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(
                set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(
                set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(
                set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')


# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods
if __name__ == '__main__':
    with open("../IMDB_crawled.json", "r") as file:
        movies = json.load(file)

    counter = 0
    preprocessed_documents = []
    for index, movie in enumerate(movies):
        counter += 1
        print(counter)
        movie = json.loads(movie)
        movie['first_page_summary'] = Preprocessor([movie['first_page_summary']]).preprocess()[0]
        movie['stars'] = Preprocessor(movie['stars']).preprocess()
        movie['genres'] = Preprocessor(movie['genres']).preprocess()
        if movie['summaries'] is not None:
            movie['summaries'] = Preprocessor(movie['summaries']).preprocess()
        else:
            continue
        if movie['synopsis'] is not None:
            movie['synopsis'] = Preprocessor(movie['synopsis']).preprocess()
        for review_index, review in enumerate(movie['reviews']):
            movie['reviews'][review_index][0] = Preprocessor([review[0]]).preprocess()[0]
        preprocessed_documents.append(movie)

    index = Index(preprocessed_documents)
    index.store_index('./index.json', None)
    index.store_index('./index.json', Indexes.GENRES.value)
    index.store_index('./index.json', Indexes.SUMMARIES.value)
    index.store_index('./index.json', Indexes.STARS.value)
    index.store_index('./index.json', Indexes.DOCUMENTS.value)


    # check loaded correctly -> for test change load index logic and add return instead of update self.index
    # index.store_index('./index.json', Indexes.GENRES.value)
    # print(index.check_if_index_loaded_correctly(Indexes.GENRES.value, index.load_index('index.json/genres.json')))

    # check indexing is good
    # index.store_index('./index.json', Indexes.SUMMARIES.value)
    # print(index.check_if_indexing_is_good(Indexes.SUMMARIES.value, 'amidst'))

    # check add/remove is correct
    # dummy_document = {
    #     'id': '101',
    #     'stars': ['tim', 'henry'],
    #     'genres': ['drama', 'crime'],
    #     'summaries': ['good']
    # }
    # index = Index([dummy_document])
    # index.check_add_remove_is_correct()
