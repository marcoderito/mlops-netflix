import requests
import os
import json
import csv

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
    assets_path = "./assets/"
    poster_folder = "posters/"
    file_name: str = movie_title.replace(" ", "_") + ".jpg"
    file_path = f"{assets_path}{poster_folder}{file_name}"
    if os.path.exists(file_path):
        #print(f"./{poster_folder}{file_name}")
        return f"./{poster_folder}{file_name}"

    config: json = json.loads(__get_config())
    movie_details: json = json.loads(__find_movie(movie_title, release_year))

    try:
        image_complete_url: str = f"{config['images']['secure_base_url']}{config['images']['poster_sizes'][4]}{movie_details['results'][0]['poster_path']}"
        response = requests.get(image_complete_url, headers=headers)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return f"./{poster_folder}{file_name}"
        else:
            return f"./{poster_folder}fail.jpeg"
    except:
        return f"./{poster_folder}fail.jpeg"




# if __name__ == "__main__":
#     file_name = "kdOXdPIgbbCHXb51tWJZ0r8kZfe.jpg"
#     url = f"https://image.tmdb.org/t/p/w500/{file_name}"
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         with open(f"./assets/posters/{file_name}", 'wb') as f:
#             f.write(response.content)
    #show_id,type,title,director,cast,country,date_added,release_year,rating,duration,listed_in,description,,,,,,,,,,,,,,
    """with open("./assets/netflix_titles.csv", "r", newline="") as movies_csv:
        titles_reader = csv.reader(movies_csv, delimiter=",")
        titles_rows = list(titles_reader)
    with open("./assets/clustered_netflix_titles.csv", "r", newline="") as clustered_netflix_titles_csv:
        clustered_reader = csv.reader(clustered_netflix_titles_csv, delimiter=",")
        clustered_rows = list(clustered_reader)
        i: int = 0
        while i < len(titles_rows):
            print(titles_rows[i][7])
            clustered_rows[i].append(titles_rows[i][7])
            print(clustered_rows[i])
            i += 1
        with open("./assets/clustered_netflix_titles.csv", "w", newline="") as clustered_netflix_titles_csv:
            clustered_writer = csv.writer(clustered_netflix_titles_csv)
            clustered_writer.writerows(clustered_rows)"""
