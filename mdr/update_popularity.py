import json
import os

# Dizionario di esempio per la popolarità degli show
example_popularity = {
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

# Funzione per inizializzare il file di popolarità
def initialize_popularity_file(popularity_file, popularity_dict):
    with open(popularity_file, 'w') as file:
        json.dump(popularity_dict, file)

# Funzione principale
def main():
    popularity_file = 'popularity.json'
    # Controlla se il file di popolarità esiste già
    if not os.path.exists(popularity_file):
        initialize_popularity_file(popularity_file, example_popularity)
        print(f"File {popularity_file} creato e inizializzato con valori di esempio.")
    else:
        print(f"File {popularity_file} esiste già. Nessuna inizializzazione necessaria.")

# Esegui la funzione main se lo script è eseguito direttamente
if __name__ == "__main__":
    main()
