import pandas
import flet as ft
import machine_learning_utils
from datetime import date


#{'sub': '061eb240-5001-70bb-465b-d8be328cee9c', 'email_verified': 'true', 'address': 'Aurisina 71', 'birthdate': '26/11/1995', 'gender': 'male', 'email': 'dream.team@outlook.it', 'username': 'cesarstef'}

class User:

    def __init__(self, username: str, birth_date: str, gender: str, given_name: str, family_name: str, page: ft.Page):
        self.username: str = username
        self.birth_date: str = birth_date
        self.gender: str = gender
        self.given_name: str = given_name
        self.family_name: str = family_name
        self.movie_dict: dict = self.__get_movie_dict(page)
        self.movie_rating_data_frame: pandas.DataFrame = self.__get_movies_rating_data_frame(page)
        self.rating: str = self.__get_rating()
        self.movie_list_data_frame = machine_learning_utils.get_movies_data_frame_filtered_by_rating(self.rating)
        self.profile: str = self.__get_profile(page)
        self.seen_show: set = self.__get_seen_show()

    def avatar(self) -> ft.CircleAvatar:
        return ft.CircleAvatar(
            foreground_image_src="",
            content=ft.Text(f"{self.given_name[0]} {self.family_name[0]}"),
            radius=50
        )

    def save_profile(self, profile: str, page: ft.Page) -> None:
        page.client_storage.set(f"{self.username}.profile", profile)

    def determine_profile(self) -> str:
        like_shows = self.movie_rating_data_frame.loc[
            (self.movie_rating_data_frame['user_id'] == self.username) & (
                        self.movie_rating_data_frame['rating'] == 'like'), 'show_id'].tolist()
        if not like_shows:
            return "Unknown Profile"
        profiles = self.movie_list_data_frame[self.movie_list_data_frame['show_id'].isin(like_shows)][
            'profile'].value_counts()
        profile_id = profiles.idxmax() if not profiles.empty else None
        print(f"profile_id: {profile_id}")
        return profile_id.__str__()

    def __get_profile(self, page: ft.Page):
        return self.determine_profile()
        '''profile = page.client_storage.get(f"{self.username}.profile")
        if profile is None or profile == "" or profile == "No profile assigned yet": #TODO: delete No profile bla bla
            profile = "Unknown Profile"
            self.save_profile(profile, page)
        return profile'''

    def __get_movie_dict(self, page: ft.Page) -> dict:
        movie_dict: dict = page.client_storage.get(f"{self.username}.movies")
        if movie_dict is None:
            movie_data_frame: pandas.DataFrame = pandas.DataFrame(columns=['user_id', 'show_id', 'rating'])
            movie_dict: dict = movie_data_frame.to_dict()
            self.save_movie_dict(movie_dict, page)
        return movie_dict

    def __get_movies_rating_data_frame(self, page: ft.Page) -> pandas.DataFrame:
        movie_dict: dict = page.client_storage.get(f"{self.username}.movies_df")
        if movie_dict is None:
            movies_rating_data_frame: pandas.DataFrame = pandas.DataFrame(columns=['user_id', 'show_id', 'rating'])
            movie_dict: dict = movies_rating_data_frame.to_dict()
            self.save_movie_dict(movie_dict, page)
        else:
            movies_rating_data_frame: pandas.DataFrame = pandas.DataFrame.from_dict(movie_dict)
        return movies_rating_data_frame

    def save_movie_dict(self, movie_dict: dict, page: ft.Page) -> None:
        page.client_storage.set(f"{self.username}.movies", movie_dict)

    def save_movies_rating_data_frame(self, movie_data_frame: pandas.DataFrame, page: ft.Page) -> None:
        page.client_storage.set(f"{self.username}.movies_df", movie_data_frame.to_dict())

    def __get_rating(self):
        today = date.today()
        birth_month: str = self.birth_date.split("/")[0]
        birth_day: str = self.birth_date.split("/")[1]
        birth_year: str = self.birth_date.split("/")[2]
        age = today.year - int(birth_year) - ((today.month, today.day) < (int(birth_month), int(birth_day)))
        if age < 18:
            return 'G'
        elif age < 21:
            return 'PG'
        elif age < 25:
            return 'PG-13'
        else:
            return 'R'

    def reset_seen_show(self):
        self.seen_show = self.__get_seen_show()
    def __get_seen_show(self):
        return set(self.movie_rating_data_frame[self.movie_rating_data_frame['user_id'] == self.username]['show_id'])

    def me_page(self, navigationBar: ft.NavigationBar) -> ft.View:
        return ft.View(
            route="/me",
            controls=[
                ft.SafeArea(
                    minimum=ft.Padding(20, 20, 20, 50),
                    content=ft.Column
                        (
                        controls=[
                            self.avatar(),
                            ft.Row(
                                controls=[
                                    ft.Text(value=self.given_name, size=70, weight=ft.FontWeight.W_900,
                                            selectable=False),
                                    ft.Text(value=self.family_name, size=70, weight=ft.FontWeight.W_900,
                                            selectable=False)
                                ],
                                adaptive=True,
                                alignment=ft.MainAxisAlignment.CENTER
                            ),
                            ft.Row(
                                controls=[
                                    ft.Text(value="Rating: ", size=50, weight=ft.FontWeight.W_600,
                                            selectable=False),
                                    ft.Text(value=self.rating, size=50, weight=ft.FontWeight.W_600,
                                            selectable=False)
                                ],
                                adaptive=True,
                                alignment=ft.MainAxisAlignment.CENTER
                            ),
                            ft.Row(
                                controls=[
                                    ft.Text(value="Profile: ", size=50, weight=ft.FontWeight.W_600,
                                            selectable=False),
                                    ft.Text(value=self.profile, size=50, weight=ft.FontWeight.W_600,
                                            selectable=False)
                                ],
                                adaptive=True,
                                alignment=ft.MainAxisAlignment.CENTER
                            ),
                        ],
                        adaptive=True,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER
                    )
                ),
                navigationBar]
        )
