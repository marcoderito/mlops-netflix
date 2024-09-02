import json
import os
import pickle

import pandas
import flet
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, classification_report

def train_model(X_train, y_train):
    print('Training model...')
    if os.path.exists("assets/model.pkl"):
        with open("assets/model.pkl", "rb") as f:
            print("Loading model...")
            return pickle.load(f)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open("assets/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained!")
    return model


def test_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)


def generate_evaluation_report(y_test, y_pred, y_pred_proba, model, X_train, y_train, tfidf_vectorizer,
                               output_file='assets/evaluation_report.pdf'):
    if not os.path.exists(os.path.dirname(output_file)):
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
            sns.heatmap(pandas.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='Blues')
            plt.title('Classification Report')
            pdf.savefig(bbox_inches='tight')
            plt.close()


def determine_profile(user_id, ratings: pandas.DataFrame, movie_data_frame: pandas.DataFrame):
    ratings.loc[:, 'show_id'] = ratings['show_id'].astype(np.int32)
    # Preprocessing dei dati: combinare tutte le colonne testuali in una singola colonna di testo
    movie_data_frame['combined_text'] = movie_data_frame[['type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description']].astype(str).apply(' '.join, axis=1)
    ratings_df = pandas.DataFrame(ratings, columns=['user_id', 'show_id', 'rating'])
    ratings_df = ratings_df.merge(movie_data_frame[['show_id', 'combined_text']],
                                  on='show_id', how='inner')
    # Trasformazione delle caratteristiche testuali usando TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(ratings_df['combined_text'])
    # Encoding della colonna 'profile'
    label_encoder = LabelEncoder()
    ratings_df['profile'] = label_encoder.fit_transform(ratings_df['combined_text'])
    # Variabili indipendenti (X) e dipendenti (y)
    X = tfidf_matrix.toarray()
    y = ratings_df['profile'].values
    # Addestramento del modello di classificazione
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    # Predizione del profilo dell'utente
    user_ratings = ratings_df[ratings_df['user_id'] == int(user_id)]
    user_tfidf_matrix = tfidf_vectorizer.transform(user_ratings['combined_text'])
    user_X = user_tfidf_matrix.toarray()
    if user_X.size == 0:
        return label_encoder.inverse_transform([np.bincount(y).argmax()])[0]
    # Predizione dei profili usando il modello addestrato
    predicted_profiles = model.predict(user_X)
    final_profile_encoded = np.bincount(predicted_profiles).argmax()
    final_profile = label_encoder.inverse_transform([final_profile_encoded])[0]
    return final_profile


def create_profiles(ratings_df, df, profiles_file="assets/profiles.csv"):
    """
    Crea un file che associa gli utenti ai loro profili in base alle recensioni date.

    :param ratings_df: DataFrame delle recensioni con colonne ['user_id', 'show_id', 'rating']
    :param df: DataFrame degli show con colonne ['show_id', 'profile', 'listed_in', 'duration', 'release_year']
    :param profiles_file: Percorso del file dove salvare i profili degli utenti
    """
    # Elenco per memorizzare i profili utente
    print("start: create_profiles")
    user_profiles = []
    # Trova profilo per ogni utente unico
    for user_id in ratings_df['user_id'].unique():
        # Filtra le recensioni dell'utente corrente
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        # Determina il profilo dell'utente usando la funzione determine_profile
        user_profile = determine_profile(user_id, user_ratings, df)
        # Aggiungi alla lista il profilo determinato
        user_profiles.append({'user_id': user_id, 'profile': user_profile})
    # Crea un DataFrame dai profili utente
    profiles_df = pandas.DataFrame(user_profiles)
    # Salva i profili degli utenti in un file CSV
    profiles_df.to_csv(profiles_file, index=False)
    print("end: create_profiles")
def load_and_initialize_reviews(predefined_file="assets/predefined_reviews.csv", ratings_file="assets/ratings.csv") -> pandas.DataFrame:
    if os.path.exists(predefined_file):
        predefined_df = pandas.read_csv(predefined_file, encoding='ISO-8859-1')
        # Rimuovere il prefisso 's' e convertire show_id in numerico
        predefined_df['show_id'] = predefined_df['show_id'].str.replace('s', '').astype(int)
        # Converti tutti i valori della colonna 'rating' in minuscolo
        predefined_df['rating'] = predefined_df['rating'].str.lower().map({'like': 1, 'dislike': 0})
        predefined_df['show_id'] = predefined_df['show_id'].astype(str)
        # Filtra solo le recensioni 'like' e 'dislike'
        predefined_df = predefined_df[predefined_df['rating'].notna()]
        predefined_df[['user_id', 'show_id', 'rating']].to_csv(ratings_file, index=False)
    else:
        raise FileNotFoundError(f"Il file {predefined_file} non esiste.")

    return predefined_df


def split_training_test(ratings_df, movie_list_dataframe, test_size=0.2):
    print("start: split_training_test")
    if os.path.exists("assets/X_train.npy") and os.path.exists("assets/X_test.npy") and os.path.exists("assets/y_train.npy") and os.path.exists("assets/y_test.npy") and os.path.exists("assets/tfidf_vectorizer.pkl"):
        print("end: split_training_test with cache")
        return np.loadtxt("assets/X_train.npy"),  np.loadtxt("assets/X_test.npy"), np.loadtxt("assets/y_test.npy"), np.loadtxt("assets/y_train.npy"), pickle.load(open("assets/tfidf_vectorizer.pkl", "rb"))

    # Rimuovere il prefisso 's' e convertire show_id in numerico
    ratings_df['show_id'] = ratings_df['show_id'].str.replace('s', '').astype(int)
    ratings_df = ratings_df.dropna(subset=['rating'])
    ratings_df['rating'] = ratings_df['rating'].astype(float)

    movie_list_dataframe['show_id'] = movie_list_dataframe['show_id'].astype(int)
    df_ratings = ratings_df[['show_id', 'rating']].reset_index(drop=True)

    # Creazione del TfidfVectorizer utilizzando i dati
    tfidf_vectorizer = create_tfidf_vectorizer(movie_list_dataframe)
    with open("assets/tfidf_vectorizer.pkl", "wb") as handle:
        pickle.dump(tfidf_vectorizer, handle)

    # Trasformazione delle caratteristiche usando il TfidfVectorizer creato
    tfidf_matrix = transform_features(movie_list_dataframe, tfidf_vectorizer)

    # Aggiungi le valutazioni al dataframe trasformato
    df_encoded = pandas.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    df_encoded = pandas.concat([movie_list_dataframe[['show_id']].reset_index(drop=True), df_encoded], axis=1)
    df_encoded = df_encoded.merge(df_ratings, on='show_id', how='left')
    # Verifica e riempi eventuali valori NaN rimanenti
    df_encoded.fillna(0, inplace=True)

    # Separare le variabili dipendenti e indipendenti
    X = df_encoded.drop(columns=['show_id', 'rating']).values
    y = df_encoded['rating'].values

    # Integrazione con SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Split dei dati in training e test
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=test_size, random_state=42)

    np.savetxt('assets/X_train.npy', X_train)
    np.savetxt('assets/X_test.npy', X_test)
    np.savetxt('assets/y_train.npy', y_train)
    np.savetxt('assets/y_test.npy', y_test)
    print("end: split_training_test")
    return X_train, X_test, y_train, y_test, tfidf_vectorizer

def is_file_empty(file_path):
    """
    Verifica se un file CSV contiene dati effettivi oltre alle intestazioni.
    """
    if not os.path.exists(file_path):
        return True

    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines) <= 1

#Funzione per caricare e clusterizzare gli show
def preprocess_data(df):
    # Considerare solo le prime 12 colonne
    df = df.iloc[:, :12]
    # Rimuovere colonne indesiderate
    columns_to_drop = [col for col in df.columns if 'Unnamed' in col]
    df = df.drop(columns=columns_to_drop)
    # Conta le righe prima dell'eliminazione
    rows_before = df.shape[0]
    # Eliminare righe con almeno un valore mancante in qualsiasi colonna
    df = df.dropna().copy()
    # Conta le righe dopo l'eliminazione
    rows_after = df.shape[0]
    # Calcola quanti righe sono state eliminate
    rows_deleted = rows_before - rows_after
    return df


def transform_features(df, tfidf_vectorizer):
    # Rimuove la colonna 'show_id'
    df = df.drop(columns=['show_id'])
    # Riempie i valori NaN e combina tutte le variabili rilevanti in un unico campo
    df.fillna('', inplace=True)
    df['combined_text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    # Sostituisce le virgole e altri separatori con spazi
    df['combined_text'] = df['combined_text'].str.replace(',', ' ').str.replace(';', ' ')
    tfidf_matrix = tfidf_vectorizer.transform(df['combined_text'])
    return tfidf_matrix


def optimal_kmeans(data, max_k=10):
    scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append((k, score))
    best_k = max(scores, key=lambda x: x[1])[0]
    return best_k


def load_and_cluster_shows(input_file, output_file) -> pandas.DataFrame:
    """
    Carica i dati degli show e li clusterizza.

    :param input_file: Percorso del file CSV di input
    :param output_file: Percorso del file CSV di output
    :return: DataFrame clusterizzato
    """
    print("start: load_cluster_shows")
    df = pandas.read_csv(input_file, encoding='ISO-8859-1')
    df['show_id'] = df['show_id'].str.replace('s', '').astype(int)
    df = preprocess_data(df)
    vectorizer = create_tfidf_vectorizer(df)
    features_scaled = transform_features(df, vectorizer)
    best_k = optimal_kmeans(features_scaled)
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    df['profile'] = kmeans.fit_predict(features_scaled)
    df.to_csv(output_file, index=False)
    print("end: load_cluster_shows")
    return df


def create_tfidf_vectorizer(df):
    # Rimuove la colonna 'show_id'
    df = df.drop(columns=['show_id'])
    # Riempie i valori NaN e combina tutte le variabili rilevanti in un unico campo
    df.fillna('', inplace=True)
    df['combined_text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    # Sostituisce le virgole e altri separatori con spazi
    df['combined_text'] = df['combined_text'].str.replace(',', ' ').str.replace(';', ' ')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vectorizer.fit(df['combined_text'])
    return tfidf_vectorizer


def analyze_and_save_to_pdf(df, ratings_file="assets/ratings.csv", output_file="assets/data_analysis.pdf"):
    if not os.path.exists(os.path.dirname(output_file)):
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
            ratings_df = pandas.read_csv(ratings_file)

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



def get_movies_data_frame_filtered_by_rating(rating: str,
                                             movies_data_frame: pandas.DataFrame = None) -> pandas.DataFrame:
    if movies_data_frame is None:
        movies_data_frame: pandas.DataFrame = pandas.read_csv("assets/clustered_netflix_titles.csv",
                                                              on_bad_lines="warn")

    rating_mapping = {'G': ['G'], 'PG': ['G', 'PG'], 'PG-13': ['G', 'PG', 'PG-13'],
                      'R': ['G', 'PG', 'PG-13', 'R', 'TV-MA']}
    print(rating_mapping[rating])
    #print(movies_data_frame[movies_data_frame['rating'].isin(rating_mapping[rating])])
    return movies_data_frame[movies_data_frame['rating'].isin(rating_mapping[rating])]


def get_popularity(page: flet.Page) -> dict:
    popularity_dict: dict = page.client_storage.get(f"movies_popularity")
    if popularity_dict is None:
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
        #TODO: chiedere a Marco domani
        #popularity_dict = predict_popularity()
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
def get_show(popularity_dict, seen_shows, movie_list_dataframe, user_profile=None, show_to_exclude=[]):
    for show_id in show_to_exclude:
        movie_list_dataframe = movie_list_dataframe.drop(movie_list_dataframe[movie_list_dataframe['show_id'] == show_id].index)
    if user_profile:
        selected_show = select_show_based_on_profile(user_profile, seen_shows, movie_list_dataframe)
        if selected_show:
            return selected_show
    for show_id, popularity in sorted(popularity_dict.items(), key=lambda x: x[1], reverse=True):
        if show_id not in seen_shows and show_id in movie_list_dataframe['show_id'].values:
            return show_id
    selected_show = select_heterogeneous_show(seen_shows, movie_list_dataframe)
    return selected_show

# Function to get an initial show for the user to rate based on popularity
def get_initial_show(popularity_dict, seen_shows, movie_list_dataframe, user_profile=None):
    for show_id, popularity in sorted(popularity_dict.items(), key=lambda x: x[1], reverse=True):
        if show_id not in seen_shows and show_id in movie_list_dataframe['show_id'].values:
            return show_id
    if user_profile:
        selected_show = select_show_based_on_profile(user_profile, seen_shows, movie_list_dataframe)
        if selected_show:
            return selected_show
    selected_show = select_heterogeneous_show(seen_shows, movie_list_dataframe)
    return selected_show


# Function to select a show based on user profile
def select_show_based_on_profile(user_profile, seen_shows, movie_list_df):
    profile_shows = movie_list_df[movie_list_df['profile'] == user_profile]
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
