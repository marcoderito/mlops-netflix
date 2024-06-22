import requests
import os
import json

base_url = "https://api.themoviedb.org"

headers = {
    "accept": "application/json",
    "Authorization": os.getenv("MOVIE_DB_AUTHORIZATION")
}


def __get_config() -> str:
    response = requests.get(base_url + "/3/configuration", headers=headers)
    return response.text


def __find_movie(movie_title: str, release_year: str) -> str:
    query_params: dict = {"query": movie_title, "year": release_year}
    response = requests.get(base_url + "/3/search/movie", params=query_params, headers=headers)
    return response.text


def get_poster(movie_title: str, release_year: str) -> str:
    config: json = json.loads(__get_config())
    movie_details: json = json.loads(__find_movie(movie_title, release_year))
    return f"{config['images']['secure_base_url']}{config['images']['poster_sizes'][4]}{movie_details['results'][0]['poster_path']}"
