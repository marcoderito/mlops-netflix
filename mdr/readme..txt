Descrizione del Programma

Scopo
	Il programma ha come scopo principale quello di fornire suggerimenti personalizzati di show televisivi agli utenti, basandosi sulle loro preferenze di visione. Per raggiungere questo obiettivo, il programma utilizza diverse tecniche di elaborazione del linguaggio naturale e di machine learning, come la 	vettorizzazione TF-IDF e la clusterizzazione K-Means, per analizzare e categorizzare gli show disponibili. Inoltre, raccoglie feedback dagli utenti attraverso un sistema di valutazione continua, aggiornando dinamicamente i profili degli utenti per migliorare la precisione dei suggerimenti.

Tecniche Utilizzate

	Controllo dell'Età: Il programma inizia verificando l'età dell'utente per filtrare gli show in base alle restrizioni di rating appropriati per quella fascia di età.
		Funzione: check_age()
		Rating: 'G', 'PG', 'PG-13', 'R'

	Filtraggio degli Show: Utilizza i dati sull'età per filtrare il dataset degli show disponibili, mostrando solo quelli adatti all'utente.
		Funzione: filter_by_rating(df, rating)

	Selezione di Show Eterogenei: Seleziona show casuali che l'utente non ha ancora visto per garantire un'esperienza di visione varia.
		Funzione: get_heterogeneous_show(df, seen_shows)

	Valutazione degli Show: Consente agli utenti di valutare gli show con 'like', 'dislike' o 'skip', raccogliendo feedback per determinare le preferenze dell'utente.
		Funzione: rate_shows(user_id, df)

	Determinazione del Profilo Utente: Analizza le valutazioni degli utenti per determinare un profilo che rappresenta le loro preferenze di visione.
		Funzione: determine_profile(user_id, ratings, df)

	Suggerimenti Basati sul Profilo: Suggerisce nuovi show agli utenti in base al loro profilo e alle valutazioni precedenti.
		Funzione: suggest_shows(df, profile, seen_shows)

	Vettorizzazione e Clusterizzazione: Utilizza il metodo TF-IDF per trasformare le descrizioni degli show in vettori e il K-Means per raggruppare questi vettori in cluster (profili) distinti.
		Strumenti:
			TfidfVectorizer per la vettorizzazione
			KMeans per la clusterizzazione

	Valutazione Continua: Permette agli utenti di continuare a valutare nuovi show, aggiornando continuamente il loro profilo per riflettere le nuove preferenze.
		Funzione: continuous_rating(user_id, df, ratings_df, ratings_file, profiles_file)

	Gestione dei Profili Utente: Carica, aggiorna e salva i profili degli utenti per garantire che le informazioni sulle preferenze siano sempre aggiornate.
		Funzioni:
			load_user_profile(user_id, profiles_file)
			update_user_profile(user_id, profile, profiles_file)

Esecuzione del Programma
	Caricamento dei Dati: Carica il dataset degli show da un file CSV (netflix_titles.csv), selezionando solo le colonne rilevanti e combinando descrizione e generi in una nuova colonna di contenuto.
	Preprocessing: Applica la vettorizzazione TF-IDF e la clusterizzazione K-Means per assegnare un profilo a ciascuno show.
	Interazione con l'Utente:
		Chiede l'ID utente e l'età.
		Filtra gli show in base al rating.
		Verifica se esistono valutazioni precedenti e carica il profilo utente.
		Se il profilo non esiste, raccoglie valutazioni iniziali per determinarlo.
		Suggerisce show basati sul profilo determinato e raccoglie nuove valutazioni in modo continuo.
	Aggiornamento dei Profili: Aggiorna e salva le valutazioni e i profili degli utenti per migliorare continuamente i suggerimenti.

Questo approccio integrato consente al programma di fornire suggerimenti personalizzati e dinamici, migliorando costantemente l'accuratezza delle raccomandazioni grazie al feedback continuo degli utenti.