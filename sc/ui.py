"""import flet
import flet as ft
from flet_core import DismissibleDismissEvent, DismissDirection

i: int = 1

safeArea: ft.SafeArea

def main(page):
    def handle_dismiss(e: DismissibleDismissEvent):
        #safeArea.controls.remove(e.control)
        print(e.direction.value.__str__())
        #if e.direction.value == DismissDirection.START_TO_END:
        if e.direction.value.__str__() == "startToEnd":
            print("Dismissed to right -> like!")
            global i
            '''safeArea.content = ft.Dismissible(
                content=ft.Card
                    (
                    width=400,
                    content=ft.Container
                        (
                        content=ft.Image
                            (
                            src=f"/{i}.jpeg",
                            fit=ft.ImageFit.FILL,
                        ),
                        padding=ft.Padding(20, 20, 20, 20),
                        alignment=ft.alignment.center
                    ),
                ),
                on_dismiss=handle_dismiss
            )'''

            i = i + 1
        else:
            print("Dismissed to left -> dislike!")
        page.update()

    page.adaptive = True

    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    '''
    page.appbar = ft.AppBar(
        leading=ft.TextButton("New", style=ft.ButtonStyle(padding=0)),
        title=ft.Text("Adaptive AppBar"),
        actions=[
            ft.IconButton(ft.cupertino_icons.ADD, style=ft.ButtonStyle(padding=0))
        ],
        bgcolor=ft.colors.with_opacity(0.04, ft.cupertino_colors.SYSTEM_BACKGROUND),
    )
    '''

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

    newDismissible = ft.Dismissible(
        content=ft.Card
            (
            width=400,
            content=ft.Container
                (
                content=ft.Image
                    (
                    src=f"/{i}.jpeg",
                    fit=ft.ImageFit.FILL,
                ),
                padding=ft.Padding(20, 20, 20, 20),
                alignment=ft.alignment.center
            ),
        ),
        on_dismiss=handle_dismiss
    )

    page.add(
        safeArea := ft.SafeArea
            (
            minimum=flet.Padding(20, 20, 20, 50),
            content=ft.Column
                (
                controls=[dismissable := ft.Dismissible
                    (
                    content=ft.Card
                        (
                        width=400,
                        content=ft.Container
                            (
                            content=ft.Image
                                (
                                src=f"/{i}.jpeg",
                                fit=ft.ImageFit.FILL,
                            ),
                            padding=ft.Padding(20, 20, 20, 20),
                            #alignment=ft.alignment.center
                        ),
                    ),
                    on_dismiss=handle_dismiss
                ), ft.Row(controls=[ft.IconButton(
                        icon=ft.icons.CANCEL_OUTLINED,
                        icon_color="red",
                        icon_size=50,
                        tooltip="Dislike",
                        adaptive=True
                    ), ft.IconButton(
                        icon=ft.icons.ROTATE_RIGHT,
                        icon_color="yellow",
                        icon_size=50,
                        tooltip="Skip",
                        adaptive=True
                    ), ft.IconButton(
                        icon=ft.icons.SKIP_NEXT_ROUNDED,
                        icon_color="green",
                        icon_size=50,
                        tooltip="Like",
                        adaptive=True,
                        #on_click=handle_dismiss(DismissibleDismissEvent(self=dismissable, direction = "startToEnd"))
                        #on_click=handle_dismiss(DismissibleDismissEvent("startToEnd"))
                    )],
                    adaptive=True,
                    alignment=ft.MainAxisAlignment.CENTER
                )],
                adaptive=True,
                #alignment=ft.alignment.center,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            adaptive=True
        )
    )


"""
import flet
from flet_core import Dismissible

#TODO: we need to add another page. Probably we can't use different files, so work with classes?

'''
import os

import flet as ft
from flet import ElevatedButton, Page, LoginEvent
from flet.auth import OAuthProvider




def main(page: Page):
    provider = OAuthProvider(
        client_id="5508c69uarjhjnlvn42tlbokgp", #os.getenv("43kpbevdh3qcu47kmn7t9ad9p4"),
        client_secret="1vs3o87chjrrc4fvonnll9da9fo6stuuk4i79gq1qn5lkng7ejqm", #os.getenv("1151aa6pasvo2e1ubkpgvu2p6tc0tgk06jjaoi6qjfm8idjrr8kn"),
        authorization_endpoint="https://mlops.auth.eu-south-1.amazoncognito.com/oauth2/authorize",
        token_endpoint="https://mlops.auth.eu-south-1.amazoncognito.com/oauth2/token",
        user_endpoint="https://mlops.auth.eu-south-1.amazoncognito.com/oauth2/userInfo",
        user_scopes=["openid"],
        user_id_fn=lambda u: u["username"],
        redirect_url="http://localhost:8080/oauth_callback",
    )

    def login_click(e):
        print("Login clicked")
        page.login(
            provider,
            redirect_to_page=True,
            on_open_authorization_url=lambda url: page.launch_url(url, web_window_name="_self"),
        )

    loginButton = ElevatedButton("Login with AWS Cognito", on_click=login_click)
    page.add(loginButton)
    
    def logout_button_click(e):
        #page.client_storage.remove(AUTH_TOKEN_KEY)
        page.logout()
        page.remove(logoutButton)
        page.add(loginButton)

    logoutButton = ElevatedButton("Logout from AWS Cognito", on_click=logout_button_click, color="red")
    def on_login(loginEvent: LoginEvent):
        print("Login event received")
        if loginEvent.error:
            raise Exception(loginEvent.error)
        print("User ID:", page.auth.user.id)
        print("Access token:", page.auth.token.access_token)
        page.remove(loginButton)
        page.add(logoutButton)

    page.on_login = on_login

ft.app(
    target=main,
    assets_dir="assets",
    view=ft.WEB_BROWSER,
    port=8080
)
'''

import flet as ft
from flet import LoginEvent, DismissibleDismissEvent, ControlEvent, Control
from flet.security import encrypt, decrypt
import pandas
import login
import like
import machine_learning_utils
import gallery
from user import User

i: int = 1

user: User = None



secret_key = "MY_APP_SECRET_KEY"  # os.getenv("MY_APP_SECRET_KEY")


def main(page: ft.Page):
    #TODO: check if here we already need the dataframe
    # Caricamento e pre-processamento dei dati
    movie_list_dataframe: pandas.DataFrame = machine_learning_utils.load_and_cluster_shows("assets/netflix_titles.csv", "assets/clustered_netflix_titles.csv")

    # Inizializza le recensioni da predefined_reviews.csv
    movie_ratings_dataframe = machine_learning_utils.load_and_initialize_reviews()

    #TODO: probably useless
    # Esecuzione dell'analisi esplorativa e statistica dei dati con output in PDF
    #machine_learning_utils.analyze_and_save_to_pdf(movie_list_dataframe)
    # Creazione dei profili da ratings.csv e clustered_netflix_titles.csv
    machine_learning_utils.create_profiles(movie_ratings_dataframe, movie_list_dataframe)

    # Separare i dati per allenamento e test
    X_train, X_test, y_train, y_test, tfidf_vectorizer = machine_learning_utils.split_training_test(movie_ratings_dataframe, movie_list_dataframe)
    # Allenare il modello
    model = machine_learning_utils.train_model(X_train, y_train)

    # Testare il modello
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("Inizio generate")
    machine_learning_utils.generate_evaluation_report(y_test, y_pred, y_pred_proba, model, X_train, y_train, tfidf_vectorizer)
    print("Fine generate")

    machine_learning_utils.get_popularity(page)
    dismissible_ref = ft.Ref[ft.Dismissible]()
    page.scroll = ft.ScrollMode.AUTO
    global user

    '''
    loginPage = ft.View(
        "/",
        [
            ft.AppBar(title=ft.Text("Flet app"), bgcolor=ft.colors.SURFACE_VARIANT),
            ft.ElevatedButton("Visit Store", on_click=lambda _: page.go("/store")),
        ],
    )
    '''
    def on_login(loginEvent: LoginEvent):
        print("Login event received")
        if loginEvent.error:
            raise Exception(loginEvent.error)
        jt = page.auth.token.to_json()
        ejt = encrypt(jt, secret_key)
        page.client_storage.set("myapp.auth_token", ejt)

    def on_logout():
        print("Logout event received")

    def  set_movie_rating(rating: str, movie_id: str, user: User, page: flet.Page):
        user_new_rating = pandas.DataFrame([{'user_id': user.username, 'show_id': movie_id, 'rating': rating}])
        user.movie_rating_data_frame = pandas.concat([user.movie_rating_data_frame, user_new_rating], ignore_index=True)

    def handle_dismiss(e: DismissibleDismissEvent):
        control = e.control
        if type(control) == ft.IconButton:
            icon_button: ft.IconButton = control
            row: ft.Row = icon_button.parent
            column: ft.Column = row.parent
            dismissible: ft.Dismissible = column.controls[0]
        else:
            dismissible: ft.Dismissible = e.control

        card = dismissible.content
        cardColumn: ft.Column = card.content
        movie_id_box: ft.Text = cardColumn.controls[0]
        movie_id: str = movie_id_box.value
        print(movie_id)

        # if e.direction.value == DismissDirection.START_TO_END:
        print(f"direction: {e.direction.value}")
        if e.direction.value.__str__() == "startToEnd":
            set_movie_rating("like", movie_id, user, page)
        if e.direction.value.__str__() == "endToStart":
            set_movie_rating("dislike", movie_id, user, page)
        user.save_movie_dict(user.movie_dict, page)
        user.save_movies_rating_data_frame(user.movie_rating_data_frame, page)
        #TODO: At this point if we skip, we just add a movie to seen_show and go to next, next time the film we pop up again
        user.seen_show.add(movie_id)
        #print(user.movie_rating_data_frame)
        #print(f"seen_show: {user.seen_show}")
        route_change(flet.RouteChangeEvent("/"))

    def handle_click(e: ControlEvent):
        dismissible_event: DismissibleDismissEvent = DismissibleDismissEvent("none")
        if e.control.data == "like":
            dismissible_event: DismissibleDismissEvent = DismissibleDismissEvent("startToEnd")
        elif e.control.data == "dislike":
            dismissible_event: DismissibleDismissEvent = DismissibleDismissEvent("endToStart")
        dismissible_event.control = e.control
        handle_dismiss(dismissible_event)

    def changed(e: ControlEvent):
        if e.data == "0":
            page.go("/me")
        if e.data == "1":
            page.go("/")
        if e.data == "2":
            page.go("/gallery")

    navigationBar: ft.NavigationBar = ft.NavigationBar(
        destinations=[
            # we should try to have here "CircleAvatar"...
            ft.NavigationDestination(icon=ft.icons.EXPLORE, label="Me"),
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
        on_change=changed,
        selected_index=1
    )

    #likePage = like.return_like_page(handle_dismiss, navigationBar)
    def route_change(route: flet.RouteChangeEvent):
        print(f"route: {route}")
        page.views.clear()
        global user
        if page.route == "/login":
            page.title = "Login"
            ejt = page.client_storage.get("myapp.auth_token")
            if ejt:
                jt = decrypt(ejt, secret_key)
                page.login(
                    login.getOAuthProvider(),
                    redirect_to_page=True,
                    on_open_authorization_url=lambda url: page.launch_url(url, web_window_name="_self"),
                    saved_token=jt
                )
            else:
                page.login(
                    login.getOAuthProvider(),
                    redirect_to_page=True,
                    on_open_authorization_url=lambda url: page.launch_url(url, web_window_name="_self"),
                )
            user = User(
                username=page.auth.user["username"],
                birth_date=page.auth.user["birthdate"],
                gender=page.auth.user["gender"],
                given_name=page.auth.user["given_name"],
                family_name=page.auth.user["family_name"],
                page=page
            )
            #global likePage
            #likePage = like.return_like_page(handle_dismiss, navigationBar, user)
            page.go("/")
        if route.route == "/":
            if user is not None:
                page.views.append(like.return_like_page(handle_dismiss, handle_click, dismissible_ref, navigationBar, user, page))
            else:
                page.go("/login")
        if page.route == "/me":
            page.views.append(user.me_page(on_logout,navigationBar))
        if page.route == "/gallery":
            page.views.append(gallery.return_gallery_page(navigationBar, user))
        page.update()

    def view_pop(view):
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop
    page.go("/login")

    page.on_login = on_login


ft.app(target=main, assets_dir="assets")
