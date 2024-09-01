import os
import json
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def check_age():
    age = int(input("Inserisci la tua età: "))
    if age < 18:
        return 'G'
    elif age < 21:
        return 'PG'
    elif age < 25:
        return 'PG-13'
    else:
        return 'R'


def analyze_and_save_to_pdf(df, ratings_file, output_file='data_analysis.pdf'):
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



# Funzione per caricare e clusterizzare gli show

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



def load_and_cluster_shows(input_file, output_file):
    """
    Carica i dati degli show e li clusterizza.

    :param input_file: Percorso del file CSV di input
    :param output_file: Percorso del file CSV di output
    :return: DataFrame clusterizzato
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
# TODO: controllare se dentro kmeans ci sono gli id

# Funzione per ottenere uno show iniziale
def get_initial_show(popularity_dict, seen_shows, df, user_profile=None):
    # Identificare il tipo di dato degli show_id nel DataFrame
    show_id_type = df['show_id'].dtype
    # Esporta il DataFrame in un file CSV

    # Primo ciclo per controllare e selezionare gli show
    for show_id, popularity in sorted(popularity_dict.items(), key=lambda x: x[1], reverse=True):
        # Convertire show_id al tipo corretto
        if show_id_type in ['int64', 'int32']:
            show_id = int(show_id)
        elif show_id_type == 'float64':
            show_id = float(show_id)
        elif show_id_type == 'str':
            show_id = str(show_id).strip()
        else:
            print(f"Tipo di dato non supportato: {show_id_type}")


    # Secondo ciclo per ulteriori controlli e selezione
    for show_id, popularity in sorted(popularity_dict.items(), key=lambda x: x[1], reverse=True):
        # Convertire show_id al tipo corretto nel secondo ciclo
        if show_id_type == 'int64':
            show_id = int(show_id)
        elif show_id_type == 'float64':
            show_id = float(show_id)
        elif show_id_type == 'str':
            show_id = str(show_id).strip()

        if show_id not in seen_shows and show_id in df['show_id'].values:
            print(f"show_id: {show_id}")
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
    profile_shows = df[df['profile'] == user_profile]
    available_shows = [show_id for show_id in profile_shows['show_id'] if show_id not in seen_shows]
    if available_shows:
        return available_shows[0]
    else:
        return None


def select_heterogeneous_show(seen_shows, df):
    genre_counts = df['listed_in'].value_counts()
    if not genre_counts.empty:
        genre = genre_counts.idxmax()
        genre_shows = df[df['listed_in'] == genre]
        available_shows = [show_id for show_id in genre_shows['show_id'] if show_id not in seen_shows]
        if available_shows:
            return available_shows[0]
    return None


# Funzione per aggiornare la popolarità degli show
def update_popularity(popularity_file, show_id):
    if os.path.exists(popularity_file):
        print(f"Dimensione del file {popularity_file}: {os.path.getsize(popularity_file)} bytes")

    if os.path.exists(popularity_file) and os.path.getsize(popularity_file) > 0:
        try:
            with open(popularity_file, 'r') as file:
                popularity_dict = json.load(file)
        except json.JSONDecodeError:
            popularity_dict = {}
    else:
        popularity_dict = {}

    # Converti show_id in stringa per garantire la compatibilità con JSON
    show_id = str(show_id)

    if show_id in popularity_dict:
        popularity_dict[show_id] += 1
    else:
        popularity_dict[show_id] = 1

    # Ordina il dizionario in base alla popolarità in ordine decrescente
    sorted_popularity = dict(sorted(popularity_dict.items(), key=lambda item: item[1], reverse=True))

    # Scrivi i dati ordinati nel file JSON
    with open(popularity_file, 'w') as file:
        json.dump(sorted_popularity, file, indent=4)


def load_and_initialize_reviews(predefined_file, ratings_file):
    if os.path.exists(predefined_file):
        predefined_df = pd.read_csv(predefined_file, encoding='ISO-8859-1')
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
def is_file_empty(file_path):
    """
    Verifica se un file CSV contiene dati effettivi oltre alle intestazioni.
    """
    if not os.path.exists(file_path):
        return True

    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines) <= 1
def split_training_test(ratings_df, df, test_size=0.2):
    # Rimuovere il prefisso 's' e convertire show_id in numerico
    ratings_df['show_id'] = ratings_df['show_id'].str.replace('s', '').astype(int)
    ratings_df = ratings_df.dropna(subset=['rating'])
    ratings_df['rating'] = ratings_df['rating'].astype(float)

    df['show_id'] = df['show_id'].astype(int)
    df_ratings = ratings_df[['show_id', 'rating']].reset_index(drop=True)

    # Creazione del TfidfVectorizer utilizzando i dati
    tfidf_vectorizer = create_tfidf_vectorizer(df)

    # Trasformazione delle caratteristiche usando il TfidfVectorizer creato
    tfidf_matrix = transform_features(df, tfidf_vectorizer)

    # Aggiungi le valutazioni al dataframe trasformato
    df_encoded = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    df_encoded = pd.concat([df[['show_id']].reset_index(drop=True), df_encoded], axis=1)
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

    return X_train, X_test, y_train, y_test, tfidf_vectorizer
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
def test_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)


# Funzione per valutare gli show
def rate_shows(user_id, df, popularity_file):
    print("Inizia la valutazione degli show")
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
        print(f"show_id: {show_id}")
        if show_id is None:
            print("Non ci sono show abbastanza valutati per determinare un profilo2. Per favore, valuta altri show.")
            break
        show = df[df['show_id'] == show_id].iloc[0]
        print(f"Titolo: {show['title']}")
        print(f"Descrizione: {show['description']}")
        print(f"Profilo: {show['profile']}")
        rating = input("Valuta lo show (like/dislike/skip): ").strip().lower()
        if rating in ['like', 'dislike']:
            ratings.append((user_id, show['show_id'], rating))
            seen_shows.add(show['show_id'])
            count += 1
            update_popularity(popularity_file, show['show_id'])
        elif rating == 'skip':
            seen_shows.add(show['show_id'])
    return ratings


# Funzione per determinare il profilo dell'utente
def determine_profile(user_id, ratings, df):
    ratings.loc[:, 'show_id'] = ratings['show_id'].astype(np.int32)

    # Preprocessing dei dati: combinare tutte le colonne testuali in una singola colonna di testo
    df['combined_text'] = df[['type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description']].astype(str).apply(' '.join, axis=1)
    ratings_df = pd.DataFrame(ratings, columns=['user_id', 'show_id', 'rating'])
    ratings_df = ratings_df.merge(df[['show_id', 'combined_text']],
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

# Funzione per suggerire show in base al profilo
def suggest_shows(df, profile, seen_shows):
    """
    Suggerisce show all'utente in base al suo profilo e agli show visti.

    :param df: DataFrame degli show con colonne ['show_id', 'profile', 'listed_in', 'duration', 'release_year']
    :param profile: Profilo dell'utente
    :param seen_shows: Set di show già visti dall'utente
    :return: DataFrame dei suggerimenti di show
    """
    # Suggerisci show basati sul profilo dell'utente
    if not seen_shows:
        suggestions = df[df['profile'] == profile]
    else:
        suggestions = df[(df['profile'] == profile) & (~df['show_id'].isin(seen_shows))]

    if suggestions.empty:
        # Se non ci sono ulteriori show nel profilo, suggerisci da altri profili
        suggestions = df[~df['show_id'].isin(seen_shows)]

    return suggestions.head(10)


# Funzione per la valutazione continua degli show
def continuous_rating(user_id, df, ratings_df, ratings_file, profiles_file, predefined_df):
    seen_shows = set(ratings_df[ratings_df['user_id'] == int(user_id)]['show_id'])
    dislike_count = 0
    dislike_threshold = 5

    while True:
        user_profile = determine_profile(user_id, ratings_df, df)
        if user_profile == "Default Profile":
            print("Non ci sono show abbastanza valutati per determinare un profilo. Per favore, valuta altri show.")
            user_ratings = rate_shows(user_id, df, 'popularity.json')
            if not user_ratings:
                break
            new_ratings_df = pd.DataFrame(user_ratings, columns=['user_id', 'show_id', 'rating'])
            ratings_df = pd.concat([ratings_df, new_ratings_df], ignore_index=True)
            ratings_df.to_csv(ratings_file, index=False)
            create_profiles(ratings_df, df, profiles_file)
        else:
            print(f"Il profilo attuale dell'utente è: {get_profile_name(user_profile, df)}")
            if dislike_count >= dislike_threshold:
                print("Suggerendo show da altri profili a causa dei dislike consecutivi.")
                suggested_shows = suggest_shows_from_other_profiles(df, user_profile, seen_shows)
                dislike_count = 0
            else:
                suggested_shows = suggest_shows(df, user_profile, seen_shows)

            if suggested_shows.empty:
                print("Non ci sono show da suggerire.")
                break

            for _, show in suggested_shows.iterrows():
                print(f"Titolo: {show['title']}")
                print(f"Descrizione: {show['description']}")
                print(f"Profilo: {show['profile']}")
                rating = input("Valuta lo show (like/dislike/skip): ").strip().lower()
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
    suggestions = df[(df['profile'] != current_profile) & (~df['show_id'].isin(seen_shows))]
    return suggestions.head(10)


def update_user_profile(user_id, profile, profiles_file):
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
    df.loc[df['show_id'] == show_id, 'profile'] = profile
    df.to_csv(clustered_file, index=False)


def get_profile_name(profile_id, df):
    profile_counts = df[df['profile'] == profile_id]['listed_in'].value_counts()
    profile_name = profile_counts.idxmax() if not profile_counts.empty else "Unknown Profile"
    return profile_name


# Funzione per preparare i dati per Nearest Neighbors
def prepare_neighbors_data(ratings_df):
    # Convertiamo le valutazioni in un formato di matrice
    user_item_matrix = ratings_df.pivot(index='user_id', columns='show_id', values='rating').fillna(0)
    return user_item_matrix


def save_reviews(df, file_path):
    df.to_csv(file_path, index=False)

def load_predefined_reviews(file_path):
    # Carica il dataset predefinito
    predefined_df = pd.read_csv(file_path)

    # Codifica delle valutazioni (1 per like, 0 per dislike)
    predefined_df['rating'] = predefined_df['rating'].str.lower().map({'like': 1, 'dislike': 0})

    return predefined_df

# Funzione per suggerire show usando Nearest Neighbors
def suggest_shows_with_neighbors(user_id, df, ratings_df, n_neighbors=5):
    user_item_matrix = prepare_neighbors_data(ratings_df)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_item_matrix.values)

    # Trova l'indice dell'utente
    user_index = list(user_item_matrix.index).index(user_id)

    # Ottieni i vicini più vicini per l'utente
    distances, indices = model_knn.kneighbors([user_item_matrix.iloc[user_index]], n_neighbors=n_neighbors + 1)

    # Trova i suggerimenti basati sui vicini
    suggestions = []
    for i in range(1, len(distances.flatten())):
        suggested_show_id = user_item_matrix.columns[indices.flatten()[i]]
        suggested_show = df[df['show_id'] == suggested_show_id].iloc[0]
        if suggested_show_id not in suggestions:
            suggestions.append(suggested_show)

    return suggestions


def filter_by_rating(df, rating):
    rating_mapping = {'G': ['G'], 'PG': ['G', 'PG'], 'PG-13': ['G', 'PG', 'PG-13'], 'R': ['G', 'PG', 'PG-13', 'R']}
    return df[df['rating'].isin(rating_mapping[rating])]


def create_profiles(ratings_df, df, profiles_file):
    """
    Crea un file che associa gli utenti ai loro profili in base alle recensioni date.

    :param ratings_df: DataFrame delle recensioni con colonne ['user_id', 'show_id', 'rating']
    :param df: DataFrame degli show con colonne ['show_id', 'profile', 'listed_in', 'duration', 'release_year']
    :param profiles_file: Percorso del file dove salvare i profili degli utenti
    """

    # Elenco per memorizzare i profili utente
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
    profiles_df = pd.DataFrame(user_profiles)



    # Salva i profili degli utenti in un file CSV
    profiles_df.to_csv(profiles_file, index=False)


def generate_evaluation_report(y_test, y_pred, y_pred_proba, model, X_train, y_train, tfidf_vectorizer,
                               output_file='evaluation_report.pdf'):
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
    Verifica se l'utente esiste nel file dei profili.

    :param user_id: ID dell'utente
    :param profiles_file: Percorso del file dei profili
    :return: True se l'utente esiste, False altrimenti
    """
    if os.path.exists(profiles_file):
        profiles_df = pd.read_csv(profiles_file)
        return user_id in profiles_df['user_id'].values
    return False

# Funzione principale
def main():


    # File paths
    netflix_titles_file = 'netflix_titles.csv'
    predefined_reviews_file = 'predefined_reviews.csv'
    ratings_file = 'ratings.csv'
    clustered_file = 'clustered_netflix_titles.csv'
    profiles_file = 'profiles.csv'

    # Caricamento e pre-processamento dei dati
    df = load_and_cluster_shows(netflix_titles_file, clustered_file)

    # Esecuzione dell'analisi esplorativa e statistica dei dati con output in PDF
    #analyze_and_save_to_pdf(df, ratings_file, 'data_analysis.pdf')

    # Inizializza le recensioni da predefined_reviews.csv
    ratings_df = load_and_initialize_reviews(predefined_reviews_file, ratings_file)

    # Creazione dei profili da ratings.csv e clustered_netflix_titles.csv
    create_profiles(ratings_df, df, profiles_file)

    # Separare i dati per allenamento e test
    X_train, X_test, y_train, y_test, tfidf_vectorizer = split_training_test(ratings_df, df)


    # Allenare il modello
    model = train_model(X_train, y_train)

    # Testare il modello
    test_model(X_test, y_test, model)
    # Testare il modello
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("Inizio generate")
    generate_evaluation_report(y_test, y_pred, y_pred_proba, model, X_train, y_train, tfidf_vectorizer)
    print("Fine generate")
    # Interazione con l'utente
    user_id = int(input("Inserisci il tuo ID utente: "))
    age_rating = check_age()

    if is_existing_user(user_id, profiles_file):
        print(f"Utente esistente trovato con ID: {user_id}")
        # Continua la valutazione con raccomandazione
        filtered_df = filter_by_rating(df, age_rating)
        continuous_rating(user_id, filtered_df, ratings_df, ratings_file, profiles_file, ratings_df)
    else:
        print(f"Nuovo utente con ID: {user_id}")
        # Profilazione iniziale del nuovo utente
        filtered_df = filter_by_rating(df, age_rating)
        initial_ratings = rate_shows(user_id, filtered_df, 'popularity.json')
        if initial_ratings:
            new_ratings_df = pd.DataFrame(initial_ratings, columns=['user_id', 'show_id', 'rating'])
            ratings_df = pd.concat([ratings_df, new_ratings_df], ignore_index=True)
            ratings_df.to_csv(ratings_file, index=False)
            create_profiles(ratings_df, df, profiles_file)
            print("Profilo creato per il nuovo utente.")
        else:
            print("Nessuna valutazione fornita per il nuovo utente.")


if __name__ == "__main__":
    main()
