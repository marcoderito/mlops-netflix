import os
import json
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, \
    classification_report, average_precision_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def check_age():
    """
    Checks the user's age and returns the appropriate content rating category.

    :return: Content rating based on age ('G', 'PG', 'PG-13', 'R')
    """
    age = int(input("Enter your age: "))
    if age < 18:
        return 'G'
    elif age < 21:
        return 'PG'
    elif age < 25:
        return 'PG-13'
    else:
        return 'R'


def analyze_and_save_to_pdf(df, ratings_file, output_file='data_analysis.pdf'):
    """
    Performs exploratory data analysis and saves the results to a PDF file.

    :param df: DataFrame containing the show data
    :param ratings_file: Path to the ratings CSV file
    :param output_file: Path to the output PDF file for saving analysis results
    """
    with PdfPages(output_file) as pdf:
        # Descriptive Statistics
        summary_stats = df.describe(include='all')
        print(summary_stats)

        # Genre Distribution (split into individual values)
        df_genres = df['listed_in'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
        plt.figure(figsize=(12, 8))
        sns.countplot(y=df_genres, order=df_genres.value_counts().index)
        plt.title('Genre Distribution')
        plt.xlabel('Count')
        plt.ylabel('Genre')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Distribution of Release Years
        plt.figure(figsize=(12, 8))
        sns.histplot(df['release_year'], bins=20, kde=True)
        plt.title('Distribution of Release Years')
        plt.xlabel('Release Year')
        plt.ylabel('Count')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Distribution of Duration
        plt.figure(figsize=(12, 8))
        sns.histplot(df['duration'].dropna().apply(lambda x: int(x.split()[0])), bins=20, kde=True)
        plt.title('Distribution of Show Durations')
        plt.xlabel('Duration (min)')
        plt.ylabel('Count')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Count of Null Values
        null_counts = df.isnull().sum()
        print(null_counts)

        # Convert 'duration' to a numeric variable
        df['duration_numeric'] = df['duration'].dropna().apply(lambda x: int(x.split()[0]))

        # Correlation Matrix
        corr_matrix = df.corr(numeric_only=True)
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Pairplot of Numeric Variables
        sns.pairplot(df[['duration_numeric', 'release_year']])
        plt.suptitle('Pairplot of Numeric Variables', y=1.02)
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Load rating data from ratings.csv file
        ratings_df = pd.read_csv(ratings_file)

        # Rating Distribution
        plt.figure(figsize=(12, 8))
        sns.countplot(x='rating', data=ratings_df)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Write descriptive statistics and null value counts to the PDF
        d = pdf.infodict()
        d['Title'] = 'Data Analysis Report'
        d['Author'] = 'Data Engineer'
        d['Subject'] = 'Exploratory Data Analysis and Statistical Analysis'

        plt.figure(figsize=(10, 8))
        plt.text(0.01, 0.05, str(summary_stats), {'fontsize': 10}, fontproperties='monospace')
        plt.title('Descriptive Statistics')
        plt.axis('off')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.text(0.01, 0.05, str(null_counts), {'fontsize': 10}, fontproperties='monospace')
        plt.title('Null Value Counts')
        plt.axis('off')
        pdf.savefig(bbox_inches='tight')
        plt.close()


def preprocess_data(df):
    """
    Preprocesses the DataFrame by selecting the first 12 columns and removing unwanted columns.

    :param df: DataFrame containing the show data
    :return: Preprocessed DataFrame
    """
    # Consider only the first 12 columns
    df = df.iloc[:, :12]

    # Remove unwanted columns
    columns_to_drop = [col for col in df.columns if 'Unnamed' in col]
    df = df.drop(columns=columns_to_drop)

    # Remove rows with at least one missing value in any column
    df = df.dropna().copy()
    return df


def create_tfidf_vectorizer(df):
    """
    Creates and fits a TF-IDF vectorizer on the combined textual data of the DataFrame.

    :param df: DataFrame containing the show data
    :return: Fitted TfidfVectorizer object
    """
    # Remove the 'show_id' column
    df = df.drop(columns=['show_id'])

    # Fill NaN values and combine all relevant variables into a single field
    df.fillna('', inplace=True)
    df.loc[:, 'combined_text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Replace commas and other separators with spaces
    df.loc[:, 'combined_text'] = df['combined_text'].str.replace(',', ' ').str.replace(';', ' ')

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vectorizer.fit(df['combined_text'])

    return tfidf_vectorizer


def transform_features(df, tfidf_vectorizer):
    """
    Transforms the features of the DataFrame using the provided TF-IDF vectorizer.

    :param df: DataFrame containing the show data
    :param tfidf_vectorizer: Fitted TfidfVectorizer object
    :return: Transformed TF-IDF matrix
    """
    # Remove the 'show_id' column
    df = df.drop(columns=['show_id'])

    # Fill NaN values and combine all relevant variables into a single field
    df.fillna('', inplace=True)
    df.loc[:, 'combined_text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Replace commas and other separators with spaces
    df.loc[:, 'combined_text'] = df['combined_text'].str.replace(',', ' ').str.replace(';', ' ')

    tfidf_matrix = tfidf_vectorizer.transform(df['combined_text'])

    return tfidf_matrix


def optimal_kmeans(data, max_k=10):
    """
    Finds the optimal number of clusters (k) for KMeans clustering using the silhouette score.

    :param data: Data to be clustered
    :param max_k: Maximum number of clusters to test
    :return: Optimal number of clusters
    """
    scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append((k, score))
    best_k = max(scores, key=lambda x: x[1])[0]
    return best_k


def load_and_cluster_shows(input_file, output_file):
    """
    Loads show data from a CSV file, preprocesses it, and performs KMeans clustering.

    :param input_file: Path to the input CSV file containing show data
    :param output_file: Path to the output CSV file to save clustered data
    :return: Clustered DataFrame
    """
    df = pd.read_csv(input_file, encoding='ISO-8859-1')
    df['show_id'] = df['show_id'].str.replace('s', '').astype(int)
    df = preprocess_data(df)
    vectorizer = create_tfidf_vectorizer(df)
    features_scaled = transform_features(df, vectorizer)
    best_k = optimal_kmeans(features_scaled)
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    df['profile'] = kmeans.fit_predict(features_scaled)
    df.to_csv(output_file, index=False)
    return df


def get_initial_show(popularity_dict, seen_shows, df, user_profile=None):
    """
    Retrieves an initial show for a user based on popularity and profile.

    :param popularity_dict: Dictionary containing show popularity data
    :param seen_shows: Set of shows already watched by the user
    :param df: DataFrame containing the show data
    :param user_profile: User's profile (optional)
    :return: Selected show ID
    """
    # Identify the data type of show_id in the DataFrame
    show_id_type = df['show_id'].dtype

    # Second loop for further checks and selection
    for show_id, popularity in sorted(popularity_dict.items(), key=lambda x: x[1], reverse=True):
        # Convert show_id to the correct type in the second loop
        if show_id_type in ['int64', 'int32']:
            show_id = int(show_id)
        elif show_id_type == 'float64':
            show_id = float(show_id)
        elif show_id_type == 'str':
            show_id = str(show_id).strip()

        if show_id not in seen_shows and show_id in df['show_id'].values:
            return show_id

    if user_profile:
        selected_show = select_show_based_on_profile(user_profile, seen_shows, df)
        if selected_show:
            print(f"selected_show1: {selected_show}")
            return selected_show
    selected_show = select_heterogeneous_show(seen_shows, df)
    print(f"selected_show2: {selected_show}")
    return selected_show


def select_show_based_on_profile(user_profile, seen_shows, df):
    """
    Selects a show based on the user's profile.

    :param user_profile: User's profile
    :param seen_shows: Set of shows already watched by the user
    :param df: DataFrame containing the show data
    :return: Selected show ID or None if no suitable show is found
    """
    profile_shows = df[df['profile'] == user_profile]
    available_shows = [show_id for show_id in profile_shows['show_id'] if show_id not in seen_shows]
    if available_shows:
        return available_shows[0]
    else:
        return None


def select_heterogeneous_show(seen_shows, df):
    """
    Selects a show based on genre distribution to provide variety.

    :param seen_shows: Set of shows already watched by the user
    :param df: DataFrame containing the show data
    :return: Selected show ID or None if no suitable show is found
    """
    genre_counts = df['listed_in'].value_counts()
    if not genre_counts.empty:
        genre = genre_counts.idxmax()
        genre_shows = df[df['listed_in'] == genre]
        available_shows = [show_id for show_id in genre_shows['show_id'] if show_id not in seen_shows]
        if available_shows:
            return available_shows[0]
    return None


def update_popularity(popularity_file, show_id):
    """
    Updates the popularity count of a show in the popularity file.

    :param popularity_file: Path to the JSON file containing popularity data
    :param show_id: ID of the show to update
    """

    if os.path.exists(popularity_file) and os.path.getsize(popularity_file) > 0:
        try:
            with open(popularity_file, 'r') as file:
                popularity_dict = json.load(file)
        except json.JSONDecodeError:
            popularity_dict = {}
    else:
        popularity_dict = {}

    # Convert show_id to string for compatibility with JSON
    show_id = str(show_id)

    if show_id in popularity_dict:
        popularity_dict[show_id] += 1
    else:
        popularity_dict[show_id] = 1

    # Sort the dictionary by popularity in descending order
    sorted_popularity = dict(sorted(popularity_dict.items(), key=lambda item: item[1], reverse=True))

    # Write the sorted data to the JSON file
    with open(popularity_file, 'w') as file:
        json.dump(sorted_popularity, file, indent=4)


def load_and_initialize_reviews(predefined_file, ratings_file):
    """
    Loads predefined reviews from a CSV file and initializes the ratings file.

    :param predefined_file: Path to the predefined reviews CSV file
    :param ratings_file: Path to the output ratings CSV file
    :return: DataFrame of predefined reviews
    """
    if os.path.exists(predefined_file):
        predefined_df = pd.read_csv(predefined_file, encoding='ISO-8859-1')
        # Remove the 's' prefix and convert show_id to numeric
        predefined_df['show_id'] = predefined_df['show_id'].str.replace('s', '').astype(int)
        # Convert all values in the 'rating' column to lowercase
        predefined_df['rating'] = predefined_df['rating'].str.lower().map({'like': 1, 'dislike': 0})
        predefined_df['show_id'] = predefined_df['show_id'].astype(str)
        # Filter only 'like' and 'dislike' reviews
        predefined_df = predefined_df[predefined_df['rating'].notna()]
        predefined_df[['user_id', 'show_id', 'rating']].to_csv(ratings_file, index=False)
    else:
        raise FileNotFoundError(f"The file {predefined_file} does not exist.")

    return predefined_df


def is_file_empty(file_path):
    """
    Checks if a CSV file contains actual data beyond the headers.

    :param file_path: Path to the CSV file
    :return: True if the file is empty or only contains headers, False otherwise
    """
    if not os.path.exists(file_path):
        return True

    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines) <= 1


def split_training_test(ratings_df, df, test_size=0.2):
    """
    Splits the data into training and testing sets, including oversampling with SMOTE.

    :param ratings_df: DataFrame of user ratings
    :param df: DataFrame containing the show data
    :param test_size: Proportion of the data to include in the test split
    :return: Training and testing sets for features and labels, fitted TfidfVectorizer
    """
    # Remove the 's' prefix and convert show_id to numeric
    ratings_df['show_id'] = ratings_df['show_id'].str.replace('s', '').astype(int)
    ratings_df = ratings_df.dropna(subset=['rating'])
    ratings_df['rating'] = ratings_df['rating'].astype(float)

    df['show_id'] = df['show_id'].astype(int)
    df_ratings = ratings_df[['show_id', 'rating']].reset_index(drop=True)

    # Create TfidfVectorizer using the data
    tfidf_vectorizer = create_tfidf_vectorizer(df)

    # Transform features using the created TfidfVectorizer
    tfidf_matrix = transform_features(df, tfidf_vectorizer)

    # Add ratings to the transformed dataframe
    df_encoded = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    df_encoded = pd.concat([df[['show_id']].reset_index(drop=True), df_encoded], axis=1)
    df_encoded = df_encoded.merge(df_ratings, on='show_id', how='left')

    # Check and fill any remaining NaN values
    df_encoded.fillna(0, inplace=True)

    # Separate dependent and independent variables
    X = df_encoded.drop(columns=['show_id', 'rating']).values
    y = df_encoded['rating'].values

    # Integration with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test, tfidf_vectorizer


def train_model(X_train, y_train):
    """
    Trains a RandomForest model on the training data.

    :param X_train: Training feature data
    :param y_train: Training labels
    :return: Trained RandomForestClassifier model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def test_model(X_test, y_test, model):
    """
    Tests the trained model on the test data and prints the classification report.

    :param X_test: Test feature data
    :param y_test: Test labels
    :param model: Trained RandomForestClassifier model
    """
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)


def rate_shows(user_id, df, popularity_file):
    """
    Allows a user to rate shows, updating the popularity file and returning the ratings.

    :param user_id: ID of the user providing ratings
    :param df: DataFrame containing the show data
    :param popularity_file: Path to the popularity JSON file
    :return: List of ratings provided by the user
    """
    print("Start rating shows")
    ratings = []
    seen_shows = set()
    count = 0
    if os.path.exists(popularity_file) and os.path.getsize(popularity_file) > 0:
        try:
            with open(popularity_file, 'r') as file:
                popularity_dict = json.load(file)
        except json.JSONDecodeError:
            popularity_dict = {}
    else:
        popularity_dict = {}
    while count < 10:
        show_id = get_initial_show(popularity_dict, seen_shows, df)
        if show_id is None:
            print("There aren't enough rated shows to determine a profile. Please rate more shows.")
            break
        show = df[df['show_id'] == show_id].iloc[0]
        print(f"Title: {show['title']}")
        print(f"Description: {show['description']}")
        rating = input("Rate the show (like/dislike/skip): ").strip().lower()
        if rating in ['like', 'dislike']:
            ratings.append((user_id, show['show_id'], rating))
            seen_shows.add(show['show_id'])
            count += 1
            update_popularity(popularity_file, show['show_id'])
        elif rating == 'skip':
            seen_shows.add(show['show_id'])
    return ratings


def determine_profile(user_id, ratings, df):
    """
    Determines the user's profile based on their ratings and show data using a Random Forest model.

    :param user_id: ID of the user
    :param ratings: DataFrame of the user's ratings
    :param df: DataFrame containing the show data
    :return: Determined user profile
    """
    # Filter the user's ratings
    user_ratings = ratings[ratings['user_id'] == int(user_id)]

    if user_ratings.empty:
        return "Default Profile"

    # Ensure the 'show_id' column is of the same type in both DataFrames
    df.loc[:, 'show_id'] = df['show_id'].astype(np.int32)
    user_ratings.loc[:, 'show_id'] = user_ratings['show_id'].astype(np.int32)

    # Combine the text of the rated shows
    df.loc[:, 'combined_text'] = df[
        ['type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration',
         'listed_in', 'description']].astype(str).apply(' '.join, axis=1)

    # Merge the rated show IDs with their respective combined texts
    ratings_df = user_ratings.merge(df[['show_id', 'combined_text', 'profile']], on='show_id', how='inner')

    if ratings_df.empty:
        return "Default Profile"

    # TF-IDF transformation
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(ratings_df['combined_text'])

    # Independent variables (X) and dependent variables (y)
    X = tfidf_matrix.toarray()
    y = ratings_df['profile'].values

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Predict the user's profile
    user_tfidf_matrix = tfidf_vectorizer.transform(ratings_df['combined_text'])
    user_X = user_tfidf_matrix.toarray()

    if user_X.size == 0:
        return "Default Profile"

    # Predict profiles using the trained model
    predicted_profiles = model.predict(user_X)
    final_profile_encoded = np.bincount(predicted_profiles).argmax()
    final_profile = ratings_df['profile'].iloc[final_profile_encoded]

    return final_profile


def suggest_shows(df, profile, seen_shows):
    """
    Suggests shows to the user based on their profile and watched shows.

    :param df: DataFrame of shows with columns ['show_id', 'profile', 'listed_in', 'duration', 'release_year']
    :param profile: User's profile
    :param seen_shows: Set of shows already watched by the user
    :return: DataFrame of show suggestions
    """
    # Suggest shows based on the user's profile
    if not seen_shows:
        suggestions = df[df['profile'] == profile]
    else:
        suggestions = df[(df['profile'] == profile) & (~df['show_id'].isin(seen_shows))]

    if suggestions.empty:
        # If no more shows are available in the profile, suggest from other profiles
        suggestions = df[~df['show_id'].isin(seen_shows)]

    return suggestions.head(10)


def continuous_rating(user_id, df, ratings_df, ratings_file, profiles_file):
    """
    Allows a user to continuously rate shows, updating their profile and suggestions.

    :param user_id: ID of the user
    :param df: DataFrame containing the show data
    :param ratings_df: DataFrame of user ratings
    :param ratings_file: Path to the ratings CSV file
    :param profiles_file: Path to the profiles CSV file
    """
    seen_shows = set(ratings_df[ratings_df['user_id'] == int(user_id)]['show_id'])
    dislike_count = 0
    dislike_threshold = 5

    while True:
        user_profile = determine_profile(user_id, ratings_df, df)
        if user_profile == "Default Profile":
            print("There aren't enough rated shows to determine a profile. Please rate more shows.")
            user_ratings = rate_shows(user_id, df, 'popularity.json')
            if not user_ratings:
                break
            new_ratings_df = pd.DataFrame(user_ratings, columns=['user_id', 'show_id', 'rating'])
            ratings_df = pd.concat([ratings_df, new_ratings_df], ignore_index=True)
            ratings_df.to_csv(ratings_file, index=False)
            create_profiles(ratings_df, df, profiles_file)
        else:
            print(f"The user's current profile is: {user_profile}")
            if dislike_count >= dislike_threshold:
                print("Suggesting shows from other profiles due to consecutive dislikes.")
                suggested_shows = suggest_shows_from_other_profiles(df, user_profile, seen_shows)
                dislike_count = 0
            else:
                suggested_shows = suggest_shows(df, user_profile, seen_shows)

            if suggested_shows.empty:
                print("There are no shows to suggest.")
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
                        dislike_count += 1
                    else:
                        dislike_count = 0
                    user_profile = determine_profile(user_id, ratings_df, df)
                    update_user_profile(user_id, user_profile, profiles_file)
                    break
                elif rating == 'skip':
                    seen_shows.add(show['show_id'])


def suggest_shows_from_other_profiles(df, current_profile, seen_shows):
    """
    Suggests shows from profiles different from the user's current profile.

    :param df: DataFrame of shows with columns ['show_id', 'profile', 'listed_in', 'duration', 'release_year']
    :param current_profile: User's current profile
    :param seen_shows: Set of shows already watched by the user
    :return: DataFrame of show suggestions
    """
    suggestions = df[(df['profile'] != current_profile) & (~df['show_id'].isin(seen_shows))]
    return suggestions.head(10)


def update_user_profile(user_id, profile, profiles_file):
    """
    Updates the user's profile in the profiles file.

    :param user_id: ID of the user
    :param profile: User's new profile
    :param profiles_file: Path to the profiles CSV file
    """
    if os.path.exists(profiles_file):
        profiles_df = pd.read_csv(profiles_file)
    else:
        profiles_df = pd.DataFrame(columns=['user_id', 'profile'])
    if user_id in profiles_df['user_id'].values:
        profiles_df.loc[profiles_df['user_id'] == int(user_id), 'profile'] = profile
    else:
        profiles_df = profiles_df.append({'user_id': int(user_id), 'profile': profile}, ignore_index=True)
    profiles_df.to_csv(profiles_file, index=False)


def update_show_profile(df, show_id, profile, clustered_file):
    """
    Updates the profile of a show in the clustered DataFrame and saves it.

    :param df: DataFrame containing the show data
    :param show_id: ID of the show to update
    :param profile: New profile to assign to the show
    :param clustered_file: Path to the clustered CSV file
    """
    df.loc[df['show_id'] == show_id, 'profile'] = profile
    df.to_csv(clustered_file, index=False)


def prepare_neighbors_data(ratings_df):
    """
    Prepares the user-item matrix for Nearest Neighbors algorithm.

    :param ratings_df: DataFrame of user ratings
    :return: User-item matrix
    """
    # Convert ratings into a matrix format
    user_item_matrix = ratings_df.pivot(index='user_id', columns='show_id', values='rating').fillna(0)
    return user_item_matrix


def save_reviews(df, file_path):
    """
    Saves the DataFrame of reviews to a CSV file.

    :param df: DataFrame containing the reviews
    :param file_path: Path to the output CSV file
    """
    df.to_csv(file_path, index=False)


def load_predefined_reviews(file_path):
    """
    Loads predefined reviews from a CSV file and encodes the ratings.

    :param file_path: Path to the predefined reviews CSV file
    :return: DataFrame of predefined reviews
    """
    # Load the predefined dataset
    predefined_df = pd.read_csv(file_path)

    # Encode ratings (1 for like, 0 for dislike)
    predefined_df['rating'] = predefined_df['rating'].str.lower().map({'like': 1, 'dislike': 0})

    return predefined_df


def suggest_shows_with_neighbors(user_id, df, ratings_df, n_neighbors=5):
    """
    Suggests shows to the user using the Nearest Neighbors algorithm.

    :param user_id: ID of the user
    :param df: DataFrame containing the show data
    :param ratings_df: DataFrame of user ratings
    :param n_neighbors: Number of neighbors to consider for suggestions
    :return: List of suggested shows
    """
    user_item_matrix = prepare_neighbors_data(ratings_df)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_item_matrix.values)

    # Find the user's index
    user_index = list(user_item_matrix.index).index(user_id)

    # Get the nearest neighbors for the user
    distances, indices = model_knn.kneighbors([user_item_matrix.iloc[user_index]], n_neighbors=n_neighbors + 1)

    # Find suggestions based on neighbors
    suggestions = []
    for i in range(1, len(distances.flatten())):
        suggested_show_id = user_item_matrix.columns[indices.flatten()[i]]
        suggested_show = df[df['show_id'] == suggested_show_id].iloc[0]
        if suggested_show_id not in suggestions:
            suggestions.append(suggested_show)

    return suggestions


def filter_by_rating(df, rating):
    """
    Filters the DataFrame by the user's content rating.

    :param df: DataFrame containing the show data
    :param rating: Content rating to filter by ('G', 'PG', 'PG-13', 'R')
    :return: Filtered DataFrame
    """
    rating_mapping = {'G': ['G'], 'PG': ['G', 'PG'], 'PG-13': ['G', 'PG', 'PG-13'], 'R': ['G', 'PG', 'PG-13', 'R']}
    return df[df['rating'].isin(rating_mapping[rating])]


def create_profiles(ratings_df, df, profiles_file):
    """
    Creates a file associating users with their profiles based on reviews.

    :param ratings_df: DataFrame of reviews with columns ['user_id', 'show_id', 'rating']
    :param df: DataFrame of shows with columns ['show_id', 'profile', 'listed_in', 'duration', 'release_year']
    :param profiles_file: Path to the file where user profiles will be saved
    """
    # List to store user profiles
    user_profiles = []

    # Find profile for each unique user
    for user_id in ratings_df['user_id'].unique():
        # Filter reviews for the current user
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        # Determine the profile of the user using the determine_profile function

        user_profile = determine_profile(user_id, user_ratings, df)

        # Add the determined profile to the list
        user_profiles.append({'user_id': user_id, 'profile': user_profile})

    # Create a DataFrame from user profiles
    profiles_df = pd.DataFrame(user_profiles)

    # Save user profiles to a CSV file
    profiles_df.to_csv(profiles_file, index=False)


def generate_evaluation_report(y_test, y_pred, y_pred_proba, model, tfidf_vectorizer,
                               output_file='evaluation_report.pdf'):
    """
    Generates a PDF report containing the evaluation metrics and visualizations for the model.

    :param y_test: True labels for the test data
    :param y_pred: Predicted labels for the test data
    :param y_pred_proba: Predicted probabilities for the test data
    :param model: Trained RandomForestClassifier model
    :param tfidf_vectorizer: Fitted TfidfVectorizer object
    :param output_file: Path to the output PDF file for saving the report
    """
    with PdfPages(output_file) as pdf:
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # ROC Curve
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Precision-Recall Curve
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f'AP = {ap:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Feature Importance
        importances = model.feature_importances_
        feature_names = tfidf_vectorizer.get_feature_names_out()
        indices = np.argsort(importances)[-10:]  # Top 10 features
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        plt.figure(figsize=(12, 10))
        sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='Blues')
        plt.title('Classification Report')
        pdf.savefig(bbox_inches='tight')
        plt.close()


def is_existing_user(user_id, profiles_file):
    """
    Checks if the user exists in the profiles file.

    :param user_id: ID of the user
    :param profiles_file: Path to the profiles file
    :return: True if the user exists, False otherwise
    """
    if os.path.exists(profiles_file):
        profiles_df = pd.read_csv(profiles_file)
        return user_id in profiles_df['user_id'].values
    return False


def main():
    """
    Main function to execute the workflow of loading data, training the model, and interacting with the user.
    """
    # File paths
    netflix_titles_file = 'netflix_titles.csv'
    predefined_reviews_file = 'predefined_reviews.csv'
    ratings_file = 'ratings.csv'
    clustered_file = 'clustered_netflix_titles.csv'
    profiles_file = 'profiles.csv'

    # Load and preprocess data
    df = load_and_cluster_shows(netflix_titles_file, clustered_file)

    # Perform exploratory and statistical analysis with PDF output
    # analyze_and_save_to_pdf(df, ratings_file, 'data_analysis.pdf')

    # Initialize reviews from predefined_reviews.csv
    ratings_df = load_and_initialize_reviews(predefined_reviews_file, ratings_file)

    # Create profiles from ratings.csv and clustered_netflix_titles.csv
    create_profiles(ratings_df, df, profiles_file)

    # Separate data for training and testing
    X_train, X_test, y_train, y_test, tfidf_vectorizer = split_training_test(ratings_df, df)

    # Train the model
    model = train_model(X_train, y_train)

    # Test the model
    test_model(X_test, y_test, model)

    # Generate evaluation report
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("Starting evaluation report generation")
    generate_evaluation_report(y_test, y_pred, y_pred_proba, model, tfidf_vectorizer)
    print("Evaluation report generation completed")

    # User interaction
    user_id = int(input("Enter your user ID: "))
    age_rating = check_age()

    if is_existing_user(user_id, profiles_file):
        print(f"Existing user found with ID: {user_id}")
        # Continue rating with recommendations
        filtered_df = filter_by_rating(df, age_rating)
        continuous_rating(user_id, filtered_df, ratings_df, ratings_file, profiles_file)
    else:
        print(f"New user with ID: {user_id}")
        # Initial profiling of the new user
        filtered_df = filter_by_rating(df, age_rating)
        initial_ratings = rate_shows(user_id, filtered_df, 'popularity.json')
        if initial_ratings:
            new_ratings_df = pd.DataFrame(initial_ratings, columns=['user_id', 'show_id', 'rating'])
            ratings_df = pd.concat([ratings_df, new_ratings_df], ignore_index=True)
            ratings_df.to_csv(ratings_file, index=False)
            create_profiles(ratings_df, df, profiles_file)
            print("Profile created for the new user.")
            # Continue with further ratings
            continuous_rating(user_id, filtered_df, ratings_df, ratings_file, profiles_file)
        else:
            print("No ratings provided for the new user.")


if __name__ == "__main__":
    main()
