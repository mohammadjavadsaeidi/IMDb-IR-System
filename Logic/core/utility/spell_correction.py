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
            shingles.add(word[i:i + k])
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
        all_shingled_words = dict()
        word_counter = dict()

        for document in all_documents:
            words = document.split()
            for word in words:
                word_lower = word.lower()
                if word_lower not in all_shingled_words:
                    all_shingled_words[word_lower] = self.shingle_word(word_lower)
                if word_lower not in word_counter:
                    word_counter[word_lower] = 0
                word_counter[word_lower] += 1

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
        word_shingles = self.shingle_word(word.lower())
        candidate_scores = []

        for candidate_word, candidate_shingles in self.all_shingled_words.items():
            jaccard = self.jaccard_score(word_shingles, candidate_shingles)
            candidate_scores.append((candidate_word, jaccard))

        candidate_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)[:5]
        max_tf = max(
            [self.word_counter[candidate[0]] for candidate in candidate_scores if candidate[0] in self.word_counter])

        ranked_candidates = []
        for candidate, jaccard in candidate_scores:
            if candidate in self.word_counter:
                normalized_tf = self.word_counter[candidate] / max_tf
                combined_score = jaccard * normalized_tf
                ranked_candidates.append((candidate, combined_score))

        ranked_candidates = sorted(ranked_candidates, key=lambda x: x[1], reverse=True)
        return [candidate[0] for candidate in ranked_candidates]

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
        corrected_words = []
        for word in query.split():
            nearest_words = self.find_nearest_words(word)
            corrected_words.append(nearest_words[0] if nearest_words else word)
        return ' '.join(corrected_words)



if __name__ == '__main__':

    # Example usage
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A brown dog chased a white cat",
        "The quick brown fox is fast"
    ]

    spell_corrector = SpellCorrection(documents)
    print(spell_corrector.spell_check("quck bron fox"))
