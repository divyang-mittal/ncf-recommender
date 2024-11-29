import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict


class Recommendation:
    def __init__(self):
        self.df = None
        self.data = None
        self.predictions = None

    def load_dataset(self):
        # Load the dataset from file which has headers on first row
        self.df = pd.read_csv("./SpotifyFeatures.csv").fillna(0)

        # Display basic info
        print(self.df.head())
        print(self.df.columns)

    def generate_fake_user_data(self):
        np.random.seed(42)

        user_ids = np.random.randint(0, 1000, size=5000)
        song_ids = np.random.choice(self.df['id'].unique(), 5000)
        ratings = np.random.randint(1, 6, size=5000)

        # get year from song id from self.df
        years = []
        acousticness = []
        danceability = []
        instrumentalness = []
        liveness = []
        speechiness = []
        energy = []
        loudness = []
        popularity = []

        for song_id in song_ids:
            year = self.df[self.df['id'] == song_id].iloc[0]['year']
            years.append(year)

            value = self.df[self.df['id'] == song_id].iloc[0]['acousticness']
            acousticness.append(value)

            value = self.df[self.df['id'] == song_id].iloc[0]['danceability']
            danceability.append(value)

            value = self.df[self.df['id'] == song_id].iloc[0]['instrumentalness']
            instrumentalness.append(value)

            value = self.df[self.df['id'] == song_id].iloc[0]['liveness']
            liveness.append(value)

            value = self.df[self.df['id'] == song_id].iloc[0]['speechiness']
            speechiness.append(value)

            value = self.df[self.df['id'] == song_id].iloc[0]['energy']
            energy.append(value)

            value = self.df[self.df['id'] == song_id].iloc[0]['loudness']
            loudness.append(value)

            value = self.df[self.df['id'] == song_id].iloc[0]['popularity']
            popularity.append(value)


        interaction_df = pd.DataFrame({
            'UserID': user_ids,
            'SongID': song_ids,
            'Rating': ratings,

        })

        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(interaction_df[['UserID', 'SongID', 'Rating']], reader)

    def train_model(self):
        trainset, testset = train_test_split(self.data, test_size=0.25)
        model = SVD()
        model.fit(trainset)

        self.predictions = model.test(testset)
        accuracy.rmse(self.predictions)

    def get_top_n_recommendations(self, n=10):
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in self.predictions:
            top_n[uid].append((iid, est))

        # Sort recommendations
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

    def print(self, song_id, rating):
        track_info = recommendation.df[recommendation.df['id'] == song_id].iloc[0]
        print(f"Track: {track_info['name']}, Artist: {track_info['artists']}, Estimated Rating: {rating:.2f}")

recommendation = Recommendation()
recommendation.load_dataset()
recommendation.generate_fake_user_data()
recommendation.train_model()

top_n_recommendations = recommendation.get_top_n_recommendations()

# Display recommendations for a sample user
sample_user = list(top_n_recommendations.keys())[0]
print(f"Recommendations for User {sample_user}:")
for song_id, rating in top_n_recommendations[sample_user]:
    recommendation.print(song_id, rating)

