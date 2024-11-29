import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


class NCFModel(tf.keras.Model):
    def __init__(self, num_users, num_songs, embedding_dim):
        super(NCFModel, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim,
                                                        embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
                                                        embeddings_initializer='he_normal')
        self.song_embedding = tf.keras.layers.Embedding(num_songs, embedding_dim,
                                                        embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
                                                        embeddings_initializer='he_normal')

        # Dense layers for learning interaction

        self.dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.dense3 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))

        self.output_layer = tf.keras.layers.Dense(1)  # No activation, continuous output for regression

    def call(self, inputs):
        user_input, song_input = inputs
        user_emb = self.user_embedding(user_input)
        song_emb = self.song_embedding(song_input)

        concat = tf.keras.layers.Concatenate()([user_emb, song_emb])
        concat = tf.keras.layers.Flatten()(concat)

        x = self.dense1(concat)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)


class UserSongInteractionModel:
    def __init__(self):
        print("UserSongInteractionModel initialized")
        self.model = None
        self.user_ids = None
        self.song_ids = None
        self.train_data = None
        self.test_data = None

    def load_dataset(self):
        # Load the dataset from file which has headers on first row
        self.df = pd.read_csv("/content/sample_data/SpotifyFeatures.csv", header=0).fillna(0)

        # Fill missing numerical values with the average value of each column
        self.df.fillna(self.df.mean(numeric_only=True), inplace=True)

        # For categorical columns, fill missing values with the most frequent value (optional)
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        # Verify if missing values are handled
        print("Missing values after processing:\n", self.df.isnull().sum())

        numerical_columns = self.df.select_dtypes(include=['number'])
        print("Numerical Columns:\n", numerical_columns)

        # print first ten rows of this df
        print(self.df.columns)

    def prepare_data(self):
        print("Preparing data")
        np.random.seed(42)

        user_ids = np.random.randint(0, 10, size=5000)
        song_ids = np.random.choice(self.df['id'].unique(), 5000)
        ratings = np.random.randint(0, 5, size=5000)

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
            value = self.df[self.df['id'] == song_id].iloc[0]
            years.append(value['year'])
            acousticness.append(value['acousticness'])
            danceability.append(value['danceability'])
            instrumentalness.append(value['instrumentalness'])
            liveness.append(value['liveness'])
            speechiness.append(value['speechiness'])
            energy.append(value['energy'])
            loudness.append(value['loudness'])
            popularity.append(value['popularity'])

        data = pd.DataFrame({
            'user_id': user_ids,
            'song_id': song_ids,
            'interaction': ratings,
            'year': years,
            'acousticness': acousticness,
            'danceability': danceability,
            'instrumentalness': instrumentalness,
            'liveness': liveness,
            'speechiness': speechiness,
            'energy': energy,
            'loudness': loudness,
            'popularity': popularity
        })

        # Convert IDs to indices for embedding
        self.user_ids = data['user_id'].unique()
        self.song_ids = data['song_id'].unique()

        user_id_to_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        song_id_to_index = {song_id: idx for idx, song_id in enumerate(self.song_ids)}

        data['user_idx'] = data['user_id'].map(user_id_to_index)
        data['song_idx'] = data['song_id'].map(song_id_to_index)

        # Train-test split
        self.train_data, self.test_data = train_test_split(data, test_size=0.2)

    def train(self):
        print("Setting up model Training")
        embedding_dim = 32
        num_users = len(self.user_ids)
        num_songs = len(self.song_ids)

        # Instantiate and compile model
        self.model = NCFModel(num_users, num_songs, embedding_dim)
        # Set custom learning rate
        learning_rate = 0.0005  # Adjust this value as needed
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        # Prepare inputs
        train_user_input = np.array(self.train_data['user_idx'])
        train_song_input = np.array(self.train_data['song_idx'])
        train_interactions = np.array(self.train_data['interaction'], dtype=np.float32)  # Use actual ratings as target

        print("Training model")
        # Train model
        self.model.fit(
            [train_user_input, train_song_input],
            train_interactions,
            epochs=20,
            batch_size=32,
            validation_split=0.1,
            shuffle=True
        )

    def evaluate(self):
        # Test set preparation
        print("Evaluating model")
        test_user_input = np.array(self.test_data['user_idx'])
        test_song_input = np.array(self.test_data['song_idx'])
        test_interactions = np.array(self.test_data['interaction']).astype(int)

        # Evaluate model
        loss, accuracy = self.model.evaluate([test_user_input, test_song_input], test_interactions)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    def recommend(self, user_idx):
        # Generate recommendations
        num_songs = len(self.song_ids)
        song_predictions = self.model.predict([np.array([user_idx] * num_songs), np.arange(num_songs)])
        recommended_songs = np.argsort(song_predictions.squeeze())[::-1][:5]

        # print the song name for each of the recommended songs
        print("Recommended Songs:")
        for song_idx in recommended_songs:
            song_id = self.song_ids[song_idx]
            song_row = self.df[self.df['id'] == song_id].iloc[0]
            song_name = song_row['name']
            artist_name = song_row['artists']
            year = song_row['year']
            popularity = song_row['popularity']
            acousticness = song_row['acousticness']
            danceability = song_row['danceability']
            instrumentalness = song_row['instrumentalness']
            liveness = song_row['liveness']
            speechiness = song_row['speechiness']
            energy = song_row['energy']
            predicted_rating = song_predictions[song_idx][0]
            print(song_name, "by", artist_name)
            print("Year:", year)
            print("Popularity:", popularity)
            print("Acousticness:", acousticness)
            print("Danceability:", danceability)
            print("Instrumentalness:", instrumentalness)
            print("Liveness:", liveness)
            print("Speechiness:", speechiness)
            print("Energy:", energy)
            print("Predicted Rating:", predicted_rating)


model = UserSongInteractionModel()
model.load_dataset()
model.prepare_data()
model.train()
model.evaluate()
model.recommend(0)