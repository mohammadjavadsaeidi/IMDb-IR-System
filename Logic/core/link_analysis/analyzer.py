import json
from Logic.core.link_analysis.graph import LinkGraph
from Logic.core.indexer.indexes_enum import Indexes
from Logic.core.indexer.index_reader import Index_reader


class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = set()
        self.authorities = set()
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            movie = json.loads(movie)
            movie_title = movie['title']
            self.graph.add_node(movie_title)
            self.authorities.add(movie_title)
            for star in movie['stars']:
                self.graph.add_node(star)
                self.graph.add_edge(star, movie_title)
                self.graph.add_edge(movie_title, star)
                self.hubs.add(star)

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            movie = json.loads(movie)
            for star in movie['stars']:
                if star in self.hubs or star in self.authorities:
                    movie_title = movie['title']
                    self.graph.add_node(movie_title)
                    self.authorities.add(movie_title)
                    self.graph.add_edge(star, movie_title)
                    self.graph.add_edge(movie_title, star)
                    self.hubs.add(star)

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        hubs = {node: 1.0 for node in self.hubs}
        authorities = {node: 1.0 for node in self.authorities}

        for _ in range(num_iteration):
            new_authorities = {node: 0.0 for node in authorities}
            for node in authorities:
                for predecessor in self.graph.get_predecessors(node):
                    new_authorities[node] += hubs[predecessor]
            norm = sum(new_authorities.values())
            for node in new_authorities:
                new_authorities[node] /= norm

            new_hubs = {node: 0.0 for node in hubs}
            for node in hubs:
                for successor in self.graph.get_successors(node):
                    new_hubs[node] += new_authorities[successor]
            norm = sum(new_hubs.values())
            for node in new_hubs:
                new_hubs[node] /= norm

            hubs, authorities = new_hubs, new_authorities

        top_actors = sorted(hubs.items(), key=lambda item: item[1], reverse=True)[:max_result]
        top_movies = sorted(authorities.items(), key=lambda item: item[1], reverse=True)[:max_result]

        return [actor for actor, score in top_actors], [movie for movie, score in top_movies]


if __name__ == "__main__":
    crawled_data_path = '/Users/snapp/PycharmProjects/IMDb-IR-System/Logic/core/IMDB_crawled.json'

    with open(crawled_data_path, 'r') as file:
        corpus = json.load(file)

    root_set = corpus[:10]

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)

    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
