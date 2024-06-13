import pandas as pd
import os
import json

# Function to check the user's age and return the appropriate age rating
def check_age():
    age = int(input("Enter your age: "))
    if age < 18:
        return 'G'
    elif age < 21:
        return 'PG'
    elif age < 25:
        return 'PG-13'
    else:
        return 'R'

# Function to filter shows based on the provided rating
def filter_by_rating(df, rating):
    rating_mapping = {'G': ['G'], 'PG': ['G', 'PG'], 'PG-13': ['G', 'PG', 'PG-13'], 'R': ['G', 'PG', 'PG-13', 'R']}
    return df[df['rating'].isin(rating_mapping[rating])]

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

# Function to update the popularity of a show
def update_popularity(popularity_file, show_id):
    if os.path.exists(popularity_file):
        with open(popularity_file, 'r') as file:
            popularity_dict = json.load(file)
    else:
        popularity_dict = {}
    if show_id in popularity_dict:
        popularity_dict[show_id] += 1
    else:
        popularity_dict[show_id] = 1
    with open(popularity_file, 'w') as file:
        json.dump(popularity_dict, file)

# Function to rate shows, returning a list of ratings
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

# Function to determine the user's profile based on their ratings
def determine_profile(user_id, ratings, df):
    ratings_df = pd.DataFrame(ratings, columns=['user_id', 'show_id', 'rating'])
    like_shows = ratings_df.loc[
        (ratings_df['user_id'] == int(user_id)) & (ratings_df['rating'] == 'like'), 'show_id'].tolist()
    if not like_shows:
        return "Unknown Profile"
    profiles = df[df['show_id'].isin(like_shows)]['profile'].value_counts()
    profile_id = profiles.idxmax() if not profiles.empty else None
    return profile_id

# Function to suggest shows based on the user's profile and seen shows
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

# Function to continuously rate shows and update user profile
def continuous_rating(user_id, df, ratings_df, ratings_file, profiles_file):
    seen_shows = set(ratings_df[ratings_df['user_id'] == int(user_id)]['show_id'])
    dislike_count = 0  # Counter for consecutive dislikes
    dislike_threshold = 2  # Threshold of dislikes to change profile

    while True:
        user_profile = determine_profile(user_id, ratings_df.values, df)
        if user_profile == "Unknown Profile":
            print("There are not enough rated shows to determine a profile. Please rate more shows.")
            user_ratings = rate_shows(user_id, df, 'popularity.json')
            if not user_ratings:
                break
            new_ratings_df = pd.DataFrame(user_ratings, columns=['user_id', 'show_id', 'rating'])
            ratings_df = pd.concat([ratings_df, new_ratings_df], ignore_index=True)
            ratings_df.to_csv(ratings_file, index=False)
            user_profile = determine_profile(user_id, ratings_df.values, df)
            if user_profile == "Unknown Profile":
                break
            update_user_profile(user_id, user_profile, profiles_file)
        else:
            print(f"The current user profile is: {get_profile_name(user_profile, df)}")
            if dislike_count >= dislike_threshold:
                print("Suggesting shows from other profiles due to consecutive dislikes.")
                suggested_shows = suggest_shows_from_other_profiles(df, user_profile, seen_shows)
                dislike_count = 0  # Reset the counter after changing profile
            else:
                suggested_shows = suggest_shows(df, user_profile, seen_shows)

            if suggested_shows.empty:
                print("No more shows to suggest for your profile.")
                break
            for _, show in suggested_shows.iterrows():
                print(f"Title: {show['title']}")
                print(f"Description: {show['description']}")
                print(f"Profile: {show['profile']}")
                rating = input("Rate the show (like/dislike/skip): ").strip().lower()
                if rating in ['like', 'dislike']:
                    new_rating = pd.DataFrame({'user_id': [user_id], 'show_id': [show['show_id']], 'rating': [rating]})
                    ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)
                    ratings_df.to_csv(ratings_file, index=False)
                    seen_shows.add(show['show_id'])
                    if rating == 'dislike':
                        dislike_count += 1  # Increment the counter for each dislike
                    else:
                        dislike_count = 0  # Reset the counter for each like
                    user_profile = determine_profile(user_id, ratings_df.values, df)
                    update_user_profile(user_id, user_profile, profiles_file)
                    break
                elif rating == 'skip':
                    seen_shows.add(show['show_id'])

# Function to suggest shows from other profiles when user dislikes current profile suggestions
def suggest_shows_from_other_profiles(df, current_profile, seen_shows):
    other_profiles = df['profile'].unique().tolist()
    other_profiles.remove(current_profile)
    suggestions = df[(df['profile'].isin(other_profiles)) & (~df['show_id'].isin(seen_shows))]
    return suggestions.head(10)

# Function to get the name of the profile based on the profile ID
def get_profile_name(profile_id, df):
    profile_counts = df[df['profile'] == profile_id]['listed_in'].value_counts()
    if not profile_counts.empty:
        return profile_counts.idxmax()
    else:
        return "Unknown Profile"

# Function to update the user's profile in the profiles file
def update_user_profile(user_id, profile, profiles_file):
    if os.path.exists(profiles_file):
        profiles_df = pd.read_csv(profiles_file)
    else:
        profiles_df = pd.DataFrame(columns=['user_id', 'profile'])
    if int(user_id) in profiles_df['user_id'].values:
        profiles_df.loc[profiles_df['user_id'] == int(user_id), 'profile'] = profile
    else:
        new_row = pd.DataFrame({'user_id': [int(user_id)], 'profile': [profile]})
        profiles_df = pd.concat([profiles_df, new_row], ignore_index=True)
    profiles_df.to_csv(profiles_file, index=False)

# Main function to run the script
def main():
    # Load the dataset of shows
    df = pd.read_csv('clustered_netflix_titles.csv')

    user_id = input("Enter your user ID: ")
    age_rating = check_age()
    filtered_df = filter_by_rating(df, age_rating)

    # Load the ratings of shows
    ratings_file = 'ratings.csv'
    profiles_file = 'profiles.csv'
    if os.path.exists(ratings_file):
        ratings_df = pd.read_csv(ratings_file)
    else:
        ratings_df = pd.DataFrame(columns=['user_id', 'show_id', 'rating'])

    continuous_rating(user_id, filtered_df, ratings_df, ratings_file, profiles_file)

if __name__ == "__main__":
    main()
