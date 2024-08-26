import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score


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


# Function to load the data and preprocess it
def load_and_preprocess_data(ratings_file_path, shows_file_path):
    ratings_df = pd.read_csv(ratings_file_path)
    shows_df = pd.read_csv(shows_file_path)

    # Merge ratings with show data
    data = ratings_df.merge(shows_df, left_on='show_id', right_on='show_id')

    # Encode categorical features
    data['rating_label'] = data['rating'].apply(lambda x: 1 if x.lower() == 'like' else 0)

    # Encode show features
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder()

    data['encoded_genre'] = label_encoder.fit_transform(data['listed_in'])
    encoded_features = one_hot_encoder.fit_transform(data[['encoded_genre']]).toarray()

    # Create feature and target sets
    X = pd.concat([pd.DataFrame(encoded_features), data[['encoded_genre']]], axis=1)
    y = data['rating_label']

    return train_test_split(X, y, test_size=0.2, random_state=42), shows_df


# Function to train the logistic regression model
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


# Function to predict user preferences
def predict_preferences(model, X_test):
    return model.predict(X_test)


# Function to suggest shows based on the model predictions
def suggest_shows_ml(shows_df, predictions, top_n=10):
    recommended_shows = shows_df.loc[predictions == 1]
    return recommended_shows.head(top_n)


# Function to plot the accuracy of recommendations
def plot_accuracy(accuracy):
    plt.figure(figsize=(8, 6))
    plt.bar(['Accuracy'], [accuracy], color='blue')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Recommendations')
    plt.show()


# Function to display a summary table of results
def display_summary(accuracy, train_size, test_size):
    summary_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Training Set Size', 'Test Set Size'],
        'Value': [accuracy, train_size, test_size]
    })
    print(summary_df)


# Main function to run the script
def main():
    ratings_file_path = 'ratings.csv'
    shows_file_path = 'clustered_netflix_titles.csv'

    # Load and preprocess data
    (X_train, X_test, y_train, y_test), shows_df = load_and_preprocess_data(ratings_file_path, shows_file_path)
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Train the model
    model = train_model(X_train, y_train)

    # Predict user preferences
    predictions = predict_preferences(model, X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.2%}")

    # Plot and display summary
    plot_accuracy(accuracy)
    display_summary(accuracy, X_train.shape[0], X_test.shape[0])


if __name__ == "__main__":
    main()
