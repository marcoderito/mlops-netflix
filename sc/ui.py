import flet
import flet as ft
from flet_core import DismissibleDismissEvent, DismissDirection


def main(page):
    def handle_dismiss(e: DismissibleDismissEvent):
        #safeArea.controls.remove(e.control)
        print(e.direction.value.__str__())
        #if e.direction.value == DismissDirection.START_TO_END:
        if e.direction.value.__str__() == "startToEnd":
            print("Dismissed to right -> like!")
            page.add(newSafeArea)
        else:
            print("Dismissed to left -> dislike!")
        page.update()

    page.adaptive = True

    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    page.appbar = ft.AppBar(
        leading=ft.TextButton("New", style=ft.ButtonStyle(padding=0)),
        title=ft.Text("Adaptive AppBar"),
        actions=[
            ft.IconButton(ft.cupertino_icons.ADD, style=ft.ButtonStyle(padding=0))
        ],
        bgcolor=ft.colors.with_opacity(0.04, ft.cupertino_colors.SYSTEM_BACKGROUND),
    )

    page.navigation_bar = ft.NavigationBar(
        destinations=[
            #we should try to have here "CircleAvatar"...
            ft.NavigationDestination(icon=ft.icons.EXPLORE, label="You"),
            ft.NavigationDestination(
                icon=ft.icons.FAVORITE_BORDER_OUTLINED,
                selected_icon=ft.icons.FAVORITE_OUTLINED,
                label="Impression"
            ),
            ft.NavigationDestination(
                icon=ft.icons.GRID_VIEW,
                selected_icon=ft.icons.GRID_VIEW_ROUNDED,
                label="Gallery",
            ),
        ],
        border=ft.Border(
            top=ft.BorderSide(color=ft.cupertino_colors.SYSTEM_GREY2, width=0)
        ),
    )

    newSafeArea = ft.SafeArea(
        minimum=flet.Padding(20, 20, 20, 50),
        content = ft.Dismissible
        (
            content=ft.Card
            (
            width=400,
            content=ft.Container
                (
                content=ft.Image
                    (
                    src=f"/imagetest.png",
                    fit=ft.ImageFit.FILL,
                ),
                padding=ft.Padding(20, 20, 20, 20),
                alignment=ft.alignment.center
                ),
            ),
            on_dismiss=handle_dismiss
        )
    )

    page.add(
        safeArea := ft.SafeArea
            (
            minimum=flet.Padding(20, 20, 20, 50),
            content=ft.Dismissible
                (
                content=ft.Card
                    (
                    width=400,
                    content=ft.Container
                        (
                        content=ft.Image
                            (
                            src=f"/imagetest.png",
                            fit=ft.ImageFit.FILL,
                        ),
                        padding=ft.Padding(20, 20, 20, 20),
                        alignment=ft.alignment.center
                    ),
                ),
                on_dismiss=handle_dismiss
            )

        )
    )


flet.app(
    target=main,
    assets_dir="assets"
)
