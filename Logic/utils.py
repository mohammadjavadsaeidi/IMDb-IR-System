from typing import Dict, List
from Logic.core.search import SearchEngine
from Logic.core.utility.spell_correction import SpellCorrection
from Logic.core.utility.snippet import Snippet
from Logic.core.indexer.indexes_enum import Indexes, Index_types
from Logic.core.utility.preprocess import Preprocessor
import json


def loads_documents_json():
    with open("/Users/snapp/PycharmProjects/IMDb-IR-System/Logic/core/indexer/index.json/documents.json", "r") as file:
        return json.load(file)


movies_dataset = loads_documents_json()
search_engine = SearchEngine()


def correct_text(text: str, all_documents: List[str]) -> str:
    """
    Correct the given query text if it is misspelled using Jacard similarity

    Paramters
    ---------
    text: str
        The query text
    all_documents : list of str
        The input documents.

    Returns
    -------
    str
        The corrected form of the given text
    """
    spell_correction_obj = SpellCorrection(all_documents)
    text = spell_correction_obj.spell_check(text)
    return text


def search(
        query: str,
        max_result_count: int,
        method: str,
        weights: list,
        should_print=False,
        preferred_genre: str = None,
        smoothing_method=None,
        alpha=0.5,
        lamda=0.5,
):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    query:
        The query text

    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25'

    weights:
        The list, containing importance weights in the search result for each of these items:
            Indexes.STARS: weights[0],
            Indexes.GENRES: weights[1],
            Indexes.SUMMARIES: weights[2],

    preferred_genre:
        A list containing preference rates for each genre. If None, the preference rates are equal.
        (You can leave it None for now)

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    weights_dic = {index: weight for index, weight in zip([Indexes.STARS, Indexes.GENRES, Indexes.SUMMARIES], weights)}
    return search_engine.search(
        query, method, weights_dic, max_results=max_result_count, safe_ranking=True, smoothing_method=smoothing_method, alpha=alpha, lamda=lamda
    )


def get_movie_by_id(id: str, movies_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get movie by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the movie

    movies_dataset: List[Dict[str, str]]
        The dataset of movies

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The movie with the given id
    """
    result = movies_dataset.get(
        id,
        {
            "Title": "This is movie's title",
            "Summary": "This is a summary",
            "URL": "https://www.imdb.com/title/tt0111161/",
            "Cast": ["Morgan Freeman", "Tim Robbins"],
            "Genres": ["Drama", "Crime"],
            "Image_URL": "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg",
        },
    )

    result["Image_URL"] = (
        "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"
        # a default picture for selected movies
    )
    result["URL"] = (
        f"https://www.imdb.com/title/{result['id']}"  # The url pattern of IMDb movies
    )
    return result


def clean_text(text: str) -> str:
    """
    Clean the given text using the preprocessor.

    Parameters
    ----------
    text: str
        The text to be cleaned.
    preprocessor: Preprocessor
        An instance of the Preprocessor class containing the preprocessing methods.

    Returns
    -------
    str
        The cleaned text.
    """
    preprocessor = Preprocessor([text])
    cleaned_text = preprocessor.preprocess()
    return cleaned_text


if __name__ == '__main__':
    print(get_movie_by_id('tt0071562', movies_dataset))
    # list_docs = []
    # for i in list(movies_dataset.values()):
    #     list_docs.append(i['title'] + ' ' + i['first_page_summary'])
    #
    # print(correct_text('spder ma', list_docs))
