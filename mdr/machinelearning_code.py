import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# Funzione per il controllo dell'età
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


# Funzione per filtrare gli show in base al rating
def filter_by_rating(df, rating):
    rating_mapping = {'G': ['G'], 'PG': ['G', 'PG'], 'PG-13': ['G', 'PG', 'PG-13'], 'R': ['G', 'PG', 'PG-13', 'R']}
    return df[df['rating'].isin(rating_mapping[rating])]


# Funzione per ottenere uno show eterogeneo
def get_heterogeneous_show(df, seen_shows):
    sampled_df = df[~df['show_id'].isin(seen_shows)].sample(frac=1).reset_index(drop=True)
    for _, row in sampled_df.iterrows():
        if row['show_id'] not in seen_shows:
            return row
    return None


# Funzione per valutare gli show
def rate_shows(user_id, df):
    ratings = []
    seen_shows = set()
    count = 0

    while count < 10:
        show = get_heterogeneous_show(df, seen_shows)
        if show is None:
            break
        print(f"Titolo: {show['title']}")
        print(f"Descrizione: {show['description']}")
        rating = input("Valuta lo show (like/dislike/skip): ").strip().lower()
        if rating in ['like', 'dislike']:
            ratings.append((user_id, show['show_id'], rating))
            seen_shows.add(show['show_id'])
            count += 1
        elif rating == 'skip':
            seen_shows.add(show['show_id'])

    return ratings


# Funzione per determinare il profilo utente
def determine_profile(user_id, ratings, df):
    like_shows = [show_id for uid, show_id, rating in ratings if rating == 'like' and uid == user_id]
    profiles = df[df['show_id'].isin(like_shows)]['profile'].value_counts()
    return profiles.idxmax() if not profiles.empty else None


# Funzione per suggerire show in base al profilo
def suggest_shows(df, profile, seen_shows):
    suggestions = df[(df['profile'] == profile) & (~df['show_id'].isin(seen_shows))]
    return suggestions.head(10)


# Funzione per la valutazione continua degli show
def continuous_rating(user_id, df, ratings_df, ratings_file, profiles_file):
    seen_shows = set(ratings_df[ratings_df['user_id'] == user_id]['show_id'])

    while True:
        user_profile = determine_profile(user_id, ratings_df.values, df)
        if user_profile is None:
            print("Non ci sono show abbastanza valutati per determinare un profilo. Per favore, valuta altri show.")
            break

        suggested_shows = suggest_shows(df, user_profile, seen_shows)

        for _, show in suggested_shows.iterrows():
            if show['show_id'] not in seen_shows:
                print(f"Titolo: {show['title']}")
                print(f"Descrizione: {show['description']}")
                rating = input("Valuta lo show (like/dislike/skip): ").strip().lower()
                if rating in ['like', 'dislike']:
                    new_rating = pd.DataFrame({'user_id': [user_id], 'show_id': [show['show_id']], 'rating': [rating]})
                    ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)
                    ratings_df.to_csv(ratings_file, index=False)
                    seen_shows.add(show['show_id'])

                    user_profile = determine_profile(user_id, ratings_df.values, df)
                    update_user_profile(user_id, user_profile, profiles_file)
                    print(f"Il tuo nuovo profilo è: {get_profile_name(user_profile, df)}")
                    break
                elif rating == 'skip':
                    seen_shows.add(show['show_id'])


# Funzione per ottenere il nome del profilo
def get_profile_name(profile_id, df):
    profile_counts = df[df['profile'] == profile_id]['listed_in'].value_counts()
    if not profile_counts.empty:
        return profile_counts.idxmax()
    else:
        return "Unknown Profile"


# Funzione per aggiornare il profilo dell'utente nel file dei profili
def update_user_profile(user_id, profile, profiles_file):
    if os.path.exists(profiles_file):
        profiles_df = pd.read_csv(profiles_file)
    else:
        profiles_df = pd.DataFrame(columns=['user_id', 'profile'])

    profiles_df = profiles_df[profiles_df['user_id'] != user_id]
    new_profile = pd.DataFrame({'user_id': [user_id], 'profile': [profile]})
    profiles_df = pd.concat([profiles_df, new_profile], ignore_index=True)
    profiles_df.to_csv(profiles_file, index=False)


# Funzione per caricare il profilo dell'utente dal file dei profili
def load_user_profile(user_id, profiles_file):
    if os.path.exists(profiles_file):
        profiles_df = pd.read_csv(profiles_file)
        user_profile = profiles_df[profiles_df['user_id'] == user_id]['profile']
        if not user_profile.empty:
            return user_profile.values[0]
    return None


# Funzione principale
def main():
    # Caricamento del dataset
    df = pd.read_csv('netflix_titles.csv', encoding='latin1')
    df = df[['show_id', 'title', 'description', 'listed_in', 'rating']]
    df['content'] = df['description'] + ' ' + df['listed_in']
    df = df.dropna(subset=['content'])

    # Vettorizzazione del contenuto
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['content'])

    # Clusterizzazione degli show in profili
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters, random_state=0)
    km.fit(tfidf_matrix)
    df['profile'] = km.labels_

    user_id = input("Inserisci il tuo ID utente: ")
    age_rating = check_age()
    filtered_df = filter_by_rating(df, age_rating)

    # Verifica se esiste il file delle valutazioni, altrimenti crealo
    ratings_file = 'user_ratings.csv'
    profiles_file = 'user_profiles.csv'
    if not os.path.exists(ratings_file):
        ratings_df = pd.DataFrame(columns=['user_id', 'show_id', 'rating'])
        ratings_df.to_csv(ratings_file, index=False)
    else:
        ratings_df = pd.read_csv(ratings_file)

    # Carica il profilo utente se esiste
    user_profile = load_user_profile(user_id, profiles_file)

    # Se il profilo non esiste, esegue la valutazione iniziale
    if user_profile is None:
        user_ratings = rate_shows(user_id, filtered_df)
        new_ratings_df = pd.DataFrame(user_ratings, columns=['user_id', 'show_id', 'rating'])
        ratings_df = pd.concat([ratings_df, new_ratings_df], ignore_index=True)
        ratings_df.to_csv(ratings_file, index=False)
        user_profile = determine_profile(user_id, ratings_df.values, filtered_df)
        update_user_profile(user_id, user_profile, profiles_file)

    print(f"Il tuo profilo è: {get_profile_name(user_profile, filtered_df)}")
    print("Suggerimenti di show per te:")
    continuous_rating(user_id, filtered_df, ratings_df, ratings_file, profiles_file)


if __name__ == "__main__":
    main()
