import flet as ft
import pandas
import pandas as pd

import user
import machine_learning_utils
import movie_db_utils

#show_id,title,description,listed_in,rating,content,profile
#s1,Dick Johnson Is Dead,"As her father nears the end of his life, filmmaker Kirsten Johnson stages his death in inventive and comical ways to help them both face the inevitable.",Documentaries,PG-13,"As her father nears the end of his life, filmmaker Kirsten Johnson stages his death in inventive and comical ways to help them both face the inevitable. Documentaries",0

def create_rating_row(handle_click) -> ft.Row:
    return ft.Row(
        controls=[
            ft.IconButton(
                icon=ft.icons.CANCEL_OUTLINED,
                icon_color="red",
                icon_size=50,
                tooltip="Dislike",
                adaptive=True,
                on_click=handle_click,
                data="dislike"
            ), ft.IconButton(
                icon=ft.icons.ROTATE_RIGHT,
                icon_color="yellow",
                icon_size=50,
                tooltip="Skip",
                adaptive=True,
                on_click=handle_click,
                data="skip"
            ), ft.IconButton(
                icon=ft.icons.SKIP_NEXT_ROUNDED,
                icon_color="green",
                icon_size=50,
                tooltip="Like",
                adaptive=True,
                on_click=handle_click,
                data="like"
            )
        ],
        adaptive=True,
        alignment=ft.MainAxisAlignment.CENTER
    )


def create_dismissible(handle_dismiss, movie_id: str, movie_title: str, movie_release_year: str, dismissible_ref: ft.Ref, user: user.User):
    #print(user.movie_rating_data_frame)
    #print(f"seen show: {user.seen_show}")
    '''for index, row in user.movie_list_data_frame.iterrows():
        if row["show_id"] in user.seen_show:
            continue
        else:
            movie_id = row["show_id"]
            movie_title = row["title"]
            break'''
    return ft.Dismissible(
        ref=dismissible_ref,
        content=ft.Card
            (
            width=400,
            content=ft.Column(
                controls=[
                    #TODO: try to put movie_id in movie_title.data
                    ft.Text(
                        value=movie_id,
                        text_align=ft.TextAlign.CENTER,
                        weight=ft.FontWeight.W_900,
                        size=30,
                        visible=False
                    ),
                    ft.Text(
                        value=movie_title,
                        text_align=ft.TextAlign.CENTER,
                        weight=ft.FontWeight.W_900,
                        size=30
                    ),
                    ft.Container(
                        content=ft.Image(
                            src=movie_db_utils.get_poster(movie_title, movie_release_year),
                            fit=ft.ImageFit.FIT_HEIGHT,#ft.ImageFit.FILL,
                            height=300
                        ),
                        padding=ft.Padding(20, 20, 20, 20),
                    )
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
        ),
        on_dismiss=handle_dismiss
    )


#TODO: we need a better name here
def post_profile_page(handle_dismiss, handle_click, dismissible_ref: ft.Ref, navigationBar: ft.NavigationBar, user: user.User, page: ft.Page) -> ft.View:
    grid_view: ft.GridView = ft.GridView(expand=True, runs_count=2, child_aspect_ratio=1)
    print(f"seen_show: {user.seen_show}")
    i: int = 0
    #for index, row in user.movie_list_data_frame.iterrows():
    show_ids: list = []
    while i < 10:
        show_id = machine_learning_utils.get_show(machine_learning_utils.get_popularity(page), user.seen_show, user.movie_list_data_frame, user.profile, show_ids)
        show_ids.append(show_id)
        movie = user.movie_list_data_frame[user.movie_list_data_frame['show_id'] == show_id].iloc[0]
        movie_id = movie["show_id"]
        movie_title = movie["title"]
        movie_release_year = movie["release_year"]
        grid_view.controls.append(
            ft.Column(
                controls=[
                    create_dismissible(handle_dismiss, movie_id, movie_title, movie_release_year, dismissible_ref, user),
                    create_rating_row(handle_click)
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
        i += 1
    i = 0 #with the for this was usefull... so better safe then sorry
    return (
        ft.View
        ("/gallery",
         [
             ft.SafeArea(
                 minimum=ft.Padding(20, 20, 20, 50),
                 #content=grid_view,
                 content=ft.Column
                 (
                     controls=[
                         ft.Text(value=f"We think that you will like these movies:",
                                 size=70, weight=ft.FontWeight.W_900,
                                 selectable=False, text_align=ft.TextAlign.CENTER),
                         grid_view
                     ],
                     adaptive=True,
                     horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                 ),
                 adaptive=True
             ),
             navigationBar
         ],
         scroll=ft.ScrollMode.AUTO
         )
    )


#TODO: we need a better name here
def base_like_page(handle_dismiss, handle_click, dismissible_ref: ft.Ref, navigationBar: ft.NavigationBar,
                   user: user.User, page: ft.Page) -> ft.View:
    movie_id = ""
    movie_title = ""
    movie_release_year = ""
    show_id = machine_learning_utils.get_show(machine_learning_utils.get_popularity(page=page), seen_shows=user.seen_show, movie_list_dataframe=user.movie_list_data_frame, user_profile=user.profile)
    movie = user.movie_list_data_frame[user.movie_list_data_frame['show_id'] == show_id].iloc[0]
    movie_id = movie["show_id"]
    movie_title = movie["title"]
    movie_release_year = movie["release_year"]
    # for index, row in user.movie_list_data_frame.iterrows():
    #     if row["show_id"] in user.seen_show:
    #         continue
    #     else:
    #         movie_id = row["show_id"]
    #         movie_title = row["title"]
    #         movie_release_year = row["release_year"]
    #         break
    return ft.View(
        "/",
        [ft.SafeArea
            (
            minimum=ft.Padding(20, 20, 20, 50),
            content=ft.Column
                (
                controls=[
                    ft.Text(value=f"Please, rate the next {10 - user.movie_rating_data_frame.__len__()} movies:",
                            size=70, weight=ft.FontWeight.W_900,
                            selectable=False, text_align=ft.TextAlign.CENTER),
                    create_dismissible(handle_dismiss, movie_id, movie_title, movie_release_year, dismissible_ref, user),
                    create_rating_row(handle_click)
                ],
                adaptive=True,
                #alignment=ft.alignment.center,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            adaptive=True
        ),
            navigationBar]
    )


def return_like_page(handle_dismiss, handle_click, dismissible_ref: ft.Ref, navigationBar: ft.NavigationBar,
                     user: user.User, page) -> ft.View:
    if user.movie_rating_data_frame.__len__() < 10:
        return base_like_page(handle_dismiss, handle_click, dismissible_ref, navigationBar, user, page)
    else:
        user.reset_seen_show()
        return post_profile_page(handle_dismiss, handle_click, dismissible_ref, navigationBar, user, page)
