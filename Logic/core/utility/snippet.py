import re


class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side
        self.stopwords = load_stopwords()

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        cleaned_query = ' '.join([word for word in query.split() if word.lower() not in self.stopwords])

        return cleaned_query

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        not_exist_words = []

        cleaned_query = self.remove_stop_words_from_query(query)

        query_tokens = cleaned_query.split()

        regex_patterns = [re.compile(r'\b' + re.escape(token) + r'\b', re.IGNORECASE) for token in query_tokens]
        print(regex_patterns)

        final_snippet = ""

        for pattern in regex_patterns:
            match = pattern.search(doc)
            if match:
                start_index = max(0, match.start() - self.number_of_words_on_each_side)
                end_index = min(len(doc), match.end() + self.number_of_words_on_each_side)
                snippet_window = doc[start_index:end_index]

                highlighted_snippet = pattern.sub(r'***\g<0>***', snippet_window)

                final_snippet += highlighted_snippet + " ... "
            else:
                not_exist_words.append(pattern.pattern)

        final_snippet = final_snippet.strip().replace(" ... ", "... ")

        return final_snippet, not_exist_words


def load_stopwords():
    """
        Load stopwords from a file.

        Returns
        ----------
        set
            A set of stopwords.
        """
    with open("/Users/snapp/PycharmProjects/IMDb-IR-System/Logic/core/stopwords.txt", 'r') as file:
        return file.read().splitlines()


if __name__ == "__main__":
    document = ("The quick brown fox jumps over the lazy dog. A brown dog chased a white cat. The quick brown fox is "
                "fast.")

    query = "quick brown redemption"

    snippet_finder = Snippet()

    snippet, not_exist_words = snippet_finder.find_snippet(document, query)

    print("Snippet:", snippet)
    print("Words not found in document:", not_exist_words)
