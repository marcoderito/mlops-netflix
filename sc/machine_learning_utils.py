import pandas
import flet


def get_movies_data_frame_filtered_by_rating(rating: str, movies_data_frame: pandas.DataFrame = None) -> pandas.DataFrame:
    if movies_data_frame is None:
        movies_data_frame: pandas.DataFrame = pandas.read_csv("assets/clustered_netflix_titles.csv", on_bad_lines="warn")

    rating_mapping = {'G': ['G'], 'PG': ['G', 'PG'], 'PG-13': ['G', 'PG', 'PG-13'], 'R': ['G', 'PG', 'PG-13', 'R', 'TV-MA']}
    print(rating_mapping[rating])
    #print(movies_data_frame[movies_data_frame['rating'].isin(rating_mapping[rating])])
    return movies_data_frame[movies_data_frame['rating'].isin(rating_mapping[rating])]

def initialize_popularity(page: flet.Page) -> dict:
    popularity_dict: dict = page.client_storage.get(f"movies_popularity")
    if popularity_dict is None:
        #TODO: this can't stay like this...
        popularity_dict = {
            's1': 5,
            's2': 3,
            's3': 8,
            's4': 2,
            's5': 10,
            's6': 7,
            's7': 6,
            's8': 4,
            's9': 9,
            's10': 1
        }
        page.client_storage.set(f"movies_popularity", popularity_dict)
    return popularity_dict



#df = movie_list_data_frame
def suggest_shows(df, profile, seen_shows):
    if not seen_shows:
        print("You have not rated any shows yet. Suggesting based on profile.")
        suggestions = df[df['profile'] == profile]
    else:
        suggestions = df[(df['profile'] == profile) & (~df['show_id'].isin(seen_shows))]
    if suggestions.empty:
        print("No more shows to suggest for your profile. Suggesting from other profiles.")
        suggestions = df[~df['show_id'].isin(seen_shows)]
    return suggestions.head(10)

# Function to select a show based on genre if user profile is not available
def select_heterogeneous_show(seen_shows, df):
    genre_counts = df['listed_in'].value_counts()
    if not genre_counts.empty:
        genre = genre_counts.idxmax()
        genre_shows = df[df['listed_in'] == genre]
        available_shows = [show_id for show_id in genre_shows['show_id'] if show_id not in seen_shows]
        if available_shows:
            return available_shows[0]
    return None

# Function to get an initial show for the user to rate based on popularity
def get_initial_show(popularity_dict, seen_shows, df, user_profile=None):
    for show_id, popularity in sorted(popularity_dict.items(), key=lambda x: x[1], reverse=True):
        if show_id not in seen_shows and show_id in df['show_id'].values:
            return show_id
    if user_profile:
        selected_show = select_show_based_on_profile(user_profile, seen_shows, df)
        if selected_show:
            return selected_show
    selected_show = select_heterogeneous_show(seen_shows, df)
    return selected_show


# Function to select a show based on user profile
def select_show_based_on_profile(user_profile, seen_shows, df):
    profile_shows = df[df['profile'] == user_profile]
    available_shows = [show_id for show_id in profile_shows['show_id'] if show_id not in seen_shows]
    if available_shows:
        return available_shows[0]
    else:
        return None

def rate_shows(user_id, df, popularity_file):
    print("Start rating shows")
    ratings = []
    seen_shows = set()
    count = 0
    if os.path.exists(popularity_file):
        with open(popularity_file, 'r') as file:
            popularity_dict = json.load(file)
    else:
        popularity_dict = {}
    while count < 10:
        show_id = get_initial_show(popularity_dict, seen_shows, df)
        if show_id is None:
            print("There are not enough rated shows to determine a profile. Please rate more shows.")
            break
        show = df[df['show_id'] == show_id].iloc[0]
        print(f"Title: {show['title']}")
        print(f"Description: {show['description']}")
        print(f"Profile: {show['profile']}")
        rating = input("Rate the show (like/dislike/skip): ").strip().lower()
        if rating in ['like', 'dislike']:
            ratings.append((user_id, show['show_id'], rating))
            seen_shows.add(show['show_id'])
            count += 1
            update_popularity(popularity_file, show['show_id'])
        elif rating == 'skip':
            seen_shows.add(show['show_id'])
    return ratings