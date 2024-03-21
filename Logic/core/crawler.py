import requests.models
from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive'
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = set()
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        id_index = 0
        for index, value in enumerate(URL.split('/')):
            if value == "title":
                id_index = index + 1
                break
        return URL.split('/')[id_index]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        with open('IMDB_crawled.json', 'w') as f:
            json.dump(list(self.crawled), f)
        pass

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = set(json.load(f))

        with open('IMDB_not_crawled.json', 'w') as f:
            self.not_crawled = deque(json.load(f))

        for link in self.crawled:
            self.added_ids.add(self.get_id_from_URL(link))

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        try:
            response = get(URL, headers=self.headers)
            if response.status_code == 200:
                return response
            else:
                print(f"Failed to crawl {URL}. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Failed to crawl {URL}. Exception: {e}")
            return None

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        response = self.crawl(self.top_250_URL)
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            htmls = soup.find_all('div', class_='sc-b0691f29-0 jbYPfh cli-children')
            for html in htmls:
                movie_id = self.get_id_from_URL(html.find('a', class_='ipc-title-link-wrapper')['href'])
                if movie_id not in self.added_ids:
                    self.added_ids.add(movie_id)
                    self.not_crawled.append(
                        'https://www.imdb.com' + html.find('a', class_='ipc-title-link-wrapper')['href'].split('?')[0])

    def extract_top_films(self, count):
        driver = webdriver.Chrome()

        url = "https://www.imdb.com/search/title/?title_type=feature"

        driver.get(url)

        try:
            unique_urls = set()
            while len(unique_urls) < count:
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//button[contains(@class, 'ipc-see-more__button')]"))
                )
                driver.execute_script("arguments[0].scrollIntoView();", load_more_button)
                time.sleep(3)

                driver.execute_script("arguments[0].click();", load_more_button)

                WebDriverWait(driver, 10).until(
                    EC.invisibility_of_element_located((By.XPATH,
                                                        "//button[contains(@class, 'ipc-see-more__button') and "
                                                        "contains(@style, 'display: block')]"))
                )

                soup = BeautifulSoup(driver.page_source, 'html.parser')
                film_elements = soup.find_all('a', class_='ipc-title-link-wrapper')
                for film_element in film_elements:
                    movie_id = self.get_id_from_URL(film_element['href'])
                    if movie_id not in self.added_ids:
                        self.added_ids.add(movie_id)
                        self.not_crawled.append(
                            'https://www.imdb.com' + film_element['href'].split('?')[
                                0])
                        unique_urls.add(movie_id)
                        if len(unique_urls) > count:
                            break
        finally:
            driver.quit()

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        self.extract_top_250()
        self.extract_top_films(800)
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            while self.not_crawled and crawled_counter < self.crawling_threshold:
                self.add_queue_lock.acquire()
                URL = self.not_crawled.popleft()
                self.add_queue_lock.release()
                self.add_list_lock.acquire()
                futures.append(executor.submit(self.crawl_page_info, URL, crawled_counter))
                self.add_list_lock.release()
                crawled_counter += 1

                if not self.not_crawled:
                    wait(futures)
                    futures = []

    def crawl_page_info(self, URL, counter):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        counter
        URL: str
            The URL of the site
        """
        print("new iteration", URL, counter)
        response = self.crawl(URL)
        review_response = self.crawl(get_review_link(URL))
        summary_response = self.crawl(get_summary_link(URL))
        if response:
            movie_info = self.get_imdb_instance()
            self.extract_movie_info(response, review_response, summary_response, movie_info, URL)
            self.crawled.add(json.dumps(movie_info))
        pass

    def extract_movie_info(self, res, review_response, summary_response, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        summary_response
        review_response
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        soup = BeautifulSoup(res.text, 'html.parser')
        review_soup = BeautifulSoup(review_response.text, 'html.parser')
        summary_soup = BeautifulSoup(summary_response.text, 'html.parser')
        movie['id'] = self.get_id_from_URL(URL)
        movie['title'] = get_title(soup)
        movie['first_page_summary'] = get_first_page_summary(soup)
        movie['release_year'] = get_release_year(soup, self.get_id_from_URL(URL))
        movie['mpaa'] = get_mpaa(soup, res)
        movie['budget'] = get_budget(soup)
        movie['gross_worldwide'] = get_gross_worldwide(soup)
        movie['rating'] = get_rating(soup)
        movie['directors'] = get_director(soup)
        movie['writers'] = get_writers(soup)
        movie['stars'] = get_stars(soup)
        movie['related_links'] = get_related_links(soup)
        movie['genres'] = get_genres(soup)
        movie['languages'] = get_languages(soup)
        movie['countries_of_origin'] = get_countries_of_origin(soup)
        movie['summaries'] = get_summary(summary_soup)
        movie['synopsis'] = get_synopsis(summary_soup)
        movie['reviews'] = get_reviews_with_scores(review_soup)

        return movie


def get_summary_link(url):
    """
    Get the link to the summary page of the movie
    Example:
    https://www.imdb.com/title/tt0111161/ is the page
    https://www.imdb.com/title/tt0111161/plotsummary is the summary page

    Parameters
    ----------
    url: str
        The URL of the site
    Returns
    ----------
    str
        The URL of the summary page
    """
    try:
        # TODO
        return url + 'plotsummary'
        pass
    except:
        print("failed to get summary link")


def get_review_link(url):
    """
    Get the link to the review page of the movie
    Example:
    https://www.imdb.com/title/tt0111161/ is the page
    https://www.imdb.com/title/tt0111161/reviews is the review page
    """
    try:
        return url + 'reviews'
        pass
    except:
        print("failed to get review link")


def get_title(soup):
    """
    Get the title of the movie from the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    str
        The title of the movie

    """

    try:
        title_element = soup.find('h1')
        if title_element:
            return title_element.text.strip()
        pass
    except:
        print("failed to get title")


def get_first_page_summary(soup):
    """
    Get the first page summary of the movie from the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    str
        The first page summary of the movie
    """
    try:
        plot_summary_element = soup.find('p', class_='sc-466bb6c-3', attrs={'data-testid': 'plot'})
        return plot_summary_element.text.strip()
        pass
    except:
        print("failed to get first page summary")


def get_director(soup):
    """
    Get the directors of the movie from the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    List[str]
        The directors of the movie
    """
    try:
        directors_section = soup.find('div', class_='ipc-metadata-list-item__content-container')
        if directors_section:
            directors = directors_section.find_all('a')
            return [director.text.strip() for director in directors]
    except:
        print("failed to get director")


def get_stars(soup):
    """
    Get the stars of the movie from the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    List[str]
        The stars of the movie
    """
    try:
        stars_section = soup.find_all('div', class_='ipc-metadata-list-item__content-container')[2]
        if stars_section:
            stars = stars_section.find_all('a')
            return [star.text.strip() for star in stars]
    except:
        print("failed to get stars")


def get_writers(soup):
    """
    Get the writers of the movie from the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    List[str]
        The writers of the movie
    """
    try:
        writers_section = soup.find_all('div', class_='ipc-metadata-list-item__content-container')[1]
        if writers_section:
            writers = writers_section.find_all('a')
            return [writer.text.strip() for writer in writers]
    except:
        print("failed to get writers")


def get_related_links(soup):
    """
    Get the related links of the movie from the More like this section of the page from the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    List[str]
        The related links of the movie
    """
    try:
        all_links = soup.find_all("a", href=True)
        return [link['href'] for link in all_links if "/title/tt" in link['href']]
        pass
    except:
        print("failed to get related links")


def get_summary(soup):
    """
    Get the summary of the movie from the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    List[str]
        The summary of the movie
    """
    try:
        summary_containers = soup.find('div', attrs={'data-testid': 'sub-section-summaries'}).find_all("div",
                                                                                                       class_="ipc-html-content-inner-div")
        summaries = []
        for container in summary_containers:
            summary_text = container.text
            summaries.append(summary_text)

        return summaries
        pass
    except:
        print("failed to get summary")


def get_synopsis(soup):
    """
    Get the synopsis of the movie from the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    List[str]
        The synopsis of the movie
    """
    try:
        synopsis_containers = soup.find('div', attrs={'data-testid': 'sub-section-synopsis'}).find_all("div",
                                                                                                       class_="ipc-html-content-inner-div")
        synopses = []
        for container in synopsis_containers:
            synopsis_text = container.text
            synopses.append(synopsis_text)

        return synopses
        pass
    except:
        print("failed to get synopsis")


def get_reviews_with_scores(soup):
    """
    Get the reviews of the movie from the soup
    reviews structure: [[review,score]]

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    List[List[str]]
        The reviews of the movie
    """
    try:
        review_containers = soup.find_all("div", class_="lister-item-content")
        reviews = []
        for container in review_containers:
            review_text = container.find("div", class_="text show-more__control").text.strip()
            rating_span = container.find("span", class_="rating-other-user-rating")
            if rating_span:
                score = rating_span.find("span").text.strip()
            else:
                score = "Not Rated"
            reviews.append([review_text, score])

        return reviews
        pass
    except:
        print("failed to get reviews")


def get_genres(soup):
    """
    Get the genres of the movie from the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    List[str]
        The genres of the movie
    """
    try:
        genre_elements = soup.find_all('span', class_='ipc-chip__text')
        return [genre.text for genre in genre_elements[:len(genre_elements) - 1]]
        pass
    except:
        print("Failed to get generes")


def get_rating(soup):
    """
    Get the rating of the movie from the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    str
        The rating of the movie
    """
    try:
        rating_span = soup.find('span', class_='sc-bde20123-1')
        return rating_span.text if rating_span else None
    except:
        print("failed to get rating")


def get_mpaa(soup, response):
    """
    Get the MPAA of the movie from the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    str
        The MPAA of the movie
    """
    try:
        mpaa_pattern = re.compile(r'contentRating":"(.*?)",')
        mpaa_match = mpaa_pattern.search(response.text)
        if mpaa_match:
            return mpaa_match.group(1)
    except:
        print("failed to get mpaa")


def get_release_year(soup, id):
    """
    Get the release year of the movie from the soup

    Parameters
    ----------
    id
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    str
        The release year of the movie
    """
    try:
        text = '/title/' + id + '/releaseinfo?ref_=tt_ov_rdat'
        year_element = soup.find('a', {'href': text})
        if year_element:
            return year_element.text.strip()
    except:
        print("failed to get release year")


def get_languages(soup):
    """
    Get the languages of the movie from the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    List[str]
        The languages of the movie
    """
    try:
        return [language.text.strip() for language in
                soup.find('li', {'data-testid': 'title-details-languages'}).find_all('a',
                                                                                     class_='ipc-metadata-list-item__list-content-item')]
    except:
        print("failed to get languages")
        return None


def get_countries_of_origin(soup):
    """
    Get the countries of origin of the movie from the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    List[str]
        The countries of origin of the movie
    """
    try:
        return [country.text.strip() for country in
                soup.find('li', {'data-testid': 'title-details-origin'}).find_all('a',
                                                                                  class_='ipc-metadata-list-item__list-content-item')]
        pass
    except:
        print("failed to get countries of origin")


def get_budget(soup):
    """
    Get the budget of the movie from box office section of the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    str
        The budget of the movie
    """
    try:
        budget_section = soup.find('li', class_='ipc-metadata-list__item',
                                   attrs={'data-testid': 'title-boxoffice-budget'})
        if budget_section:
            budget_span = budget_section.find('span', class_='ipc-metadata-list-item__list-content-item')
            if budget_span:
                return budget_span.text
        pass
    except:
        print("failed to get budget")


def get_gross_worldwide(soup):
    """
    Get the gross worldwide of the movie from box office section of the soup

    Parameters
    ----------
    soup: BeautifulSoup
        The soup of the page
    Returns
    ----------
    str
        The gross worldwide of the movie
    """
    try:
        gross_worldwide_section = soup.find('li', class_='ipc-metadata-list__item',
                                            attrs={'data-testid': 'title-boxoffice-cumulativeworldwidegross'})
        if gross_worldwide_section:
            gross_worldwide_span = gross_worldwide_section.find('span',
                                                                class_='ipc-metadata-list-item__list-content-item')
            if gross_worldwide_span:
                return gross_worldwide_span.text
        pass
    except:
        print("failed to get gross worldwide")


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=1000)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
