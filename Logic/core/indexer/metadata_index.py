from Logic.core.indexer.index_reader import Index_reader
from Logic.core.indexer.indexes_enum import Indexes, Index_types
import json


class Metadata_index:
    def __init__(self, path='/Users/snapp/PycharmProjects/IMDb-IR-System/Logic/core/indexer/index.json'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        self.index_reader = Index_reader(path, Indexes.DOCUMENTS)
        self.documents = self.read_documents()
        self.metadata_index = self.create_metadata_index()
        self.store_metadata_index(path)

    def read_documents(self):
        """
        Reads the documents.
        
        """
        return self.index_reader.get_index()

    def create_metadata_index(self):
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['average_document_length'] = {
            'stars': self.get_average_document_field_length('stars'),
            'genres': self.get_average_document_field_length('genres'),
            'summaries': self.get_average_document_field_length('summaries')
        }
        metadata_index['document_count'] = len(self.documents)

        return metadata_index

    def get_average_document_field_length(self, where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """
        total_length = sum(len(doc.get(where, [])) for doc in self.documents.values())
        return total_length / len(self.documents) if len(self.documents) > 0 else 0

    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path = path + '/' + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '.json'
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)


if __name__ == "__main__":
    meta_index = Metadata_index()
