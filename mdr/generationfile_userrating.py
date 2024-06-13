import pandas as pd
import random


def generate_user_ratings(netflix_titles_path, user_ratings_path, num_users=400):
    # Load the Netflix titles dataset
    netflix_titles_df = pd.read_csv(netflix_titles_path, encoding='latin1')

    # Filter necessary columns
    netflix_titles_df = netflix_titles_df[['show_id', 'title', 'description', 'listed_in', 'rating']]

    # Create a list of users
    user_ids = range(1, num_users + 1)

    # Initialize a list to hold user ratings
    user_ratings = []

    # Simulate user ratings
    for user_id in user_ids:
        liked_genres = set()
        for _ in range(random.randint(5, 15)):  # Each user rates between 5 and 15 shows
            show = netflix_titles_df.sample(1).iloc[0]
            show_id = show['show_id']
            genre = show['listed_in']

            if genre in liked_genres or random.random() < 0.7:
                rating = 'like'
                liked_genres.add(genre)
            else:
                rating = 'dislike'

            user_ratings.append([user_id, show_id, rating])

    # Create a DataFrame for user ratings
    user_ratings_df = pd.DataFrame(user_ratings, columns=['user_id', 'show_id', 'rating'])

    # Save the user ratings to CSV
    user_ratings_df.to_csv(user_ratings_path, index=False)

    return user_ratings_df


def main():
    netflix_titles_path = 'netflix_titles.csv'
    user_ratings_path = 'user_ratings.csv'
    user_ratings_df = generate_user_ratings(netflix_titles_path, user_ratings_path)
    print(user_ratings_df.head())


if __name__ == "__main__":
    main()
