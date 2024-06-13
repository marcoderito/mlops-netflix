import pandas as pd
import os


# Funzione per determinare il profilo utente
def determine_profile(user_id, ratings, shows_df):
    liked_shows = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] == 'like')]['show_id']
    if liked_shows.empty:
        return None

    liked_profiles = shows_df[shows_df['show_id'].isin(liked_shows)]['profile']
    if liked_profiles.empty:
        return None

    most_common_profile = liked_profiles.value_counts().idxmax()
    return most_common_profile


# Funzione per aggiornare il file dei profili utenti
def update_user_profiles(ratings_file, shows_file, profiles_file):
    # Carica le valutazioni degli utenti
    ratings_df = pd.read_csv(ratings_file)

    # Carica le informazioni sugli show
    shows_df = pd.read_csv(shows_file)

    # Crea un DataFrame vuoto per i profili degli utenti se non esiste già
    if os.path.exists(profiles_file):
        profiles_df = pd.read_csv(profiles_file)
    else:
        profiles_df = pd.DataFrame(columns=['user_id', 'profile'])

    # Ottieni l'elenco degli utenti unici
    user_ids = ratings_df['user_id'].unique()

    for user_id in user_ids:
        # Determina il profilo dell'utente
        profile = determine_profile(user_id, ratings_df, shows_df)
        if profile is not None:
            # Aggiorna o aggiungi il profilo dell'utente nel DataFrame dei profili
            profiles_df = profiles_df[profiles_df['user_id'] != user_id]
            new_profile = pd.DataFrame({'user_id': [user_id], 'profile': [profile]})
            profiles_df = pd.concat([profiles_df, new_profile], ignore_index=True)

    # Salva il DataFrame dei profili aggiornato
    profiles_df.to_csv(profiles_file, index=False)
    print("User profiles updated successfully.")


def main():
    # Specifica i percorsi dei file
    ratings_file = 'user_ratings.csv'
    shows_file = 'clustered_netflix_titles.csv'
    profiles_file = 'user_profiles.csv'

    # Aggiorna i profili degli utenti
    update_user_profiles(ratings_file, shows_file, profiles_file)


if __name__ == "__main__":
    main()
