import hashlib
import numpy as np
import itertools
import random
import json


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set()
        for i in range(len(document) - k + 1):
            shingle = document[i:i + k]
            shingles.add(shingle)
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        num_docs = len(self.documents)
        num_shingles = len(set().union(*[self.shingle_document(doc) for doc in self.documents]))

        characteristic_matrix = np.zeros((num_docs, num_shingles), dtype=int)

        for i, doc in enumerate(self.documents):
            shingles = self.shingle_document(doc)
            for j, shingle in enumerate(shingles):
                characteristic_matrix[i, j] = 1 if shingle in doc else 0

        return characteristic_matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        characteristic_matrix = self.build_characteristic_matrix()
        num_docs, num_shingles = characteristic_matrix.shape

        hash_funcs = [hashlib.md5() for _ in range(self.num_hashes)]

        min_hash_signatures = np.full((self.num_hashes, num_docs), np.inf)

        for i in range(num_docs):
            for j in range(num_shingles):
                if characteristic_matrix[i, j] == 1:
                    for k in range(self.num_hashes):
                        hash_funcs[k].update(str(j).encode())
                        hash_val = hash_funcs[k].digest()

                        min_hash_signatures[k, i] = min(min_hash_signatures[k, i], int.from_bytes(hash_val, "big"))

        return min_hash_signatures

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        num_docs = signature.shape[1]

        buckets = {}

        for b in range(bands):
            band_hash_vals = {}

            for doc_id in range(num_docs):
                band_signature = signature[b * rows_per_band: (b + 1) * rows_per_band, doc_id]
                band_hash = hash(tuple(band_signature))

                if band_hash not in band_hash_vals:
                    band_hash_vals[band_hash] = []

                band_hash_vals[band_hash].append(doc_id)
            for bucket_id, docs in band_hash_vals.items():
                if len(docs) > 1:
                    if bucket_id not in buckets:
                        buckets[bucket_id] = []
                    buckets[bucket_id].extend(docs)

        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        min_hash_signatures = self.min_hash_signature()
        buckets = self.lsh_buckets(min_hash_signatures)
        return buckets

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        return intersection / union if union != 0 else 0

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)


if __name__ == "__main__":
    with open("/Users/snapp/PycharmProjects/IMDb-IR-System/Logic/core/IMDB_crawled.json", "r") as file:
        data = json.load(file)

    with open("/Users/snapp/PycharmProjects/IMDb-IR-System/Logic/core/LSHFakeData.json", "r") as file:
        fake_data = json.load(file)

    summaries = []
    for movie in data:
        movie = json.loads(movie)
        if movie["summaries"] is not None:
            summaries.append(' '.join(movie["summaries"]))

    for fake_movie in fake_data:
        if fake_movie["summaries"] is not None:
            summaries.append(' '.join(fake_movie["summaries"]))

    minhash_lsh = MinHashLSH(summaries, num_hashes=20)

    buckets = minhash_lsh.perform_lsh()

    minhash_lsh.jaccard_similarity_test(buckets, summaries)
