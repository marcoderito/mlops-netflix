import flet as ft
import pandas as pd

import user

import movie_db_utils




def return_gallery_page(navigationBar: ft.NavigationBar, user: user.User) -> ft.View:
    grid_view: ft.GridView = ft.GridView(expand=True, runs_count=5, child_aspect_ratio=2)
    for index, row in user.movie_rating_data_frame.iterrows():
        #movie_list_mask = user.movie_list_data_frame['show_id'] == row["show_id"]+
        #TODO: change this fucking name...
        something: pd.DataFrame = user.movie_list_data_frame[user.movie_list_data_frame["show_id"] == row["show_id"]]
        if row["rating"] == "like":
            rating_color = "green"
        else:
            rating_color = "red"
        grid_view.controls.append(
            ft.Card(
                content=ft.Column(
                    controls=[
                        ft.Text(
                            value=something["title"].values[0],
                            text_align=ft.TextAlign.CENTER,
                            weight=ft.FontWeight.W_900,
                            size=10,
                            data=something["show_id"].values[0]
                        ),
                        ft.Container(
                            content=ft.Image(
                                src=movie_db_utils.get_poster(something["title"].values[0], something["release_year"].values[0]),
                                #fit=ft.ImageFit.CONTAIN,
                            ),
                            width=100,
                            height=80,
                            #padding=ft.Padding(20, 20, 20, 20),
                            alignment=ft.alignment.center
                        ),
                        ft.Text(
                            value=f"{row['rating'].capitalize()}",
                            text_align=ft.TextAlign.CENTER,
                            color=rating_color,
                            weight=ft.FontWeight.W_900,
                            size=10,
                            data=something["show_id"]
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                adaptive=True,
                clip_behavior= ft.ClipBehavior.ANTI_ALIAS_WITH_SAVE_LAYER
            )
        )
    return (
        ft.View
        ("/gallery",
    [
                ft.SafeArea(
                    minimum=ft.Padding(20, 20, 20, 50),
                    content=grid_view,
                    adaptive=True
                ),
                navigationBar
            ],
         scroll=ft.ScrollMode.AUTO
        )
    )


