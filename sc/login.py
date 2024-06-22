from flet.auth import OAuthProvider
import os


def getOAuthProvider() -> OAuthProvider:
    return OAuthProvider(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),

        authorization_endpoint=os.getenv("AUTH_URL"),
        token_endpoint=os.getenv("TOKEN_URL"),
        user_endpoint=os.getenv("USER_URL"),
        user_scopes=["openid"],
        user_id_fn=lambda u: u["username"],  #to change, based on your IdP access_token
        redirect_url=os.getenv("REDIRECT_URL")
    )


'''
def logoutClick(e):
    # page.client_storage.remove(AUTH_TOKEN_KEY)
    page.logout()
    page.remove(logoutButton)
    page.add(loginButton)



loginButton = ElevatedButton("Login with AWS Cognito", on_click=loginClick)
logoutButton = ElevatedButton("Logout from AWS Cognito", on_click=logoutClick, color="red")




def loginHandler(loginEvent: LoginEvent):
    if loginEvent.error:
        raise Exception(loginEvent.error)


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
