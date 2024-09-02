import flet as ft
import pandas as pd

import user
import gallery
import movie_db_utils


# Handles empty movie rating data frame without errors
def test_handles_empty_movie_rating_data_frame(self, mocker):
    # Mocking the necessary components
    mock_navigation_bar = mocker.Mock(spec=ft.NavigationBar)
    mock_user = mocker.Mock(spec=user.User)
    mock_user.movie_rating_data_frame = pd.DataFrame(columns=['user_id', 'show_id', 'rating'])
    mock_user.movie_list_data_frame = pd.DataFrame(columns=['show_id', 'title', 'release_year'])

    # Call the function under test
    result = gallery.return_gallery_page(mock_navigation_bar, mock_user)

    # Assertions
    assert isinstance(result, ft.View)
    assert len(result.controls[0].content.controls) == 0  # No movie cards should be present

# Generates a gallery view with a grid of movie cards
def test_generates_gallery_view_with_movie_cards(self, mocker):
    # Mocking the necessary components
    mock_navigation_bar = mocker.Mock(spec=ft.NavigationBar)
    mock_user = mocker.Mock(spec=user.User)
    mock_user.movie_rating_data_frame = pd.DataFrame({
        'user_id': ['user1', 'user1'],
        'show_id': ['show1', 'show2'],
        'rating': ['like', 'dislike']
    })
    mock_user.movie_list_data_frame = pd.DataFrame({
        'show_id': ['show1', 'show2'],
        'title': ['Movie 1', 'Movie 2'],
        'release_year': ['2020', '2021']
    })

    # Mocking the get_poster function
    mocker.patch('movie_db_utils.get_poster', return_value='path/to/poster.jpg')

    # Call the function under test
    result = gallery.return_gallery_page(mock_navigation_bar, mock_user)

    # Assertions
    assert isinstance(result, ft.View)
    assert len(result.controls[0].content.controls) == 2  # Two movie cards
    assert result.controls[0].content.controls[0].content.controls[0].value == 'Movie 1'
    assert result.controls[0].content.controls[1].content.controls[0].value == 'Movie 2'