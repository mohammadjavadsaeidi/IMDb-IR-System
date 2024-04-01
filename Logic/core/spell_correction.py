class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()
        for i in range(len(word) - k + 1):
            shingle = word[i:i + k]
            shingles.add(shingle)
        return shingles

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        return intersection / union if union != 0 else 0

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = {}
        word_counter = {}
        for document in all_documents:
            words = document.split()
            for word in words:
                shingles = self.shingle_word(word)
                all_shingled_words[word] = shingles
                word_counter[word] = word_counter.get(word, 0) + 1
        return all_shingled_words, word_counter

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : str
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = []
        shingles = self.shingle_word(word)
        max_tf = max(self.word_counter.values())
        for candidate, candidate_shingles in self.all_shingled_words.items():
            score = self.jaccard_score(shingles, candidate_shingles)
            tf_normalized = self.word_counter.get(candidate, 0) / max_tf
            weighted_score = score * tf_normalized
            top5_candidates.append((candidate, weighted_score))
        top5_candidates.sort(key=lambda x: x[1], reverse=True)
        return [candidate[0] for candidate in top5_candidates[:5]]

    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : str
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ""
        words = query.split()
        for word in words:
            corrected_word = self.find_nearest_words(word)
            final_result += corrected_word[0] if corrected_word else word
            final_result += " "
        return final_result.strip()


if __name__ == "__main__":
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A brown dog chased a white cat",
        "The quick brown fox is fast"
    ]

    spell_corrector = SpellCorrection(documents)

    misspelled_query = "quik broen foxx"
    corrected_query = spell_corrector.spell_check(misspelled_query)
    print("Misspelled Query:", misspelled_query)
    print("Corrected Query:", corrected_query)