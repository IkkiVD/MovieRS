from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
import keras

global_model = keras.saving.load_model('data/movie_recommendation_model.keras')
movies_df = pd.read_csv('data/movies.csv')
user_enc = joblib.load('data/user_enc.pkl')
item_enc = joblib.load('data/item_enc.pkl')


def rebuild_model(n_users, n_movies, old_model=None):
    n_factors = 150

    # Input layers
    user_input = keras.layers.Input(shape=(1,))
    movie_input = keras.layers.Input(shape=(1,))

    # Embedding layers
    user_embedding = keras.layers.Embedding(n_users, n_factors, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))(user_input)
    movie_embedding = keras.layers.Embedding(n_movies, n_factors, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))(movie_input)

    # Reshape embeddings
    user_vec = keras.layers.Reshape((n_factors,))(user_embedding)
    movie_vec = keras.layers.Reshape((n_factors,))(movie_embedding)

    # Concatenate embeddings
    x = keras.layers.Concatenate()([user_vec, movie_vec])
    x = keras.layers.Dropout(0.05)(x)

    # Dense layers
    x = keras.layers.Dense(32, kernel_initializer='he_normal', activation='relu')(x)
    x = keras.layers.Dropout(0.05)(x)
    x = keras.layers.Dense(16, kernel_initializer='he_normal', activation='relu')(x)
    x = keras.layers.Dropout(0.05)(x)

    # Output layer
    output = keras.layers.Dense(1, activation='linear')(x)

    # Define the model
    new_model = keras.models.Model(inputs=[user_input, movie_input], outputs=output)

    # Compile the model
    new_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Transfer weights from the old model if available
    if old_model:
        try:
            new_model.get_layer('embedding').set_weights(old_model.get_layer('embedding').get_weights())
            new_model.get_layer('embedding_1').set_weights(old_model.get_layer('embedding_1').get_weights())
        except ValueError:
            pass  # Ignore if dimensions don't match

    return new_model

def retrain_model():
    try:
        # Reload the updated ratings
        ratings_file = 'data/ratings.csv'
        data_ratings = pd.read_csv(ratings_file)
        merged_dataset = pd.merge(movies_df, data_ratings, how='inner', on='movieId')
        refined_dataset = merged_dataset.groupby(by=['userId', 'title'], as_index=False).agg({"rating": "mean"})

        # Update the encoders with all user and movie IDs
        user_enc.fit(refined_dataset['userId'].to_numpy())
        item_enc.fit(refined_dataset['title'].to_numpy())

        joblib.dump(user_enc, 'data/user_enc.pkl')
        joblib.dump(item_enc, 'data/item_enc.pkl')
        
        # Re-encode user and movie IDs
        refined_dataset['user'] = user_enc.transform(refined_dataset['userId'].to_numpy())
        refined_dataset['movie'] = item_enc.transform(refined_dataset['title'].to_numpy())
        refined_dataset['rating'] = refined_dataset['rating'].values.astype(np.float32)

        # Update the number of users and movies
        n_users = len(user_enc.classes_)
        n_movies = len(item_enc.classes_)

        # Rebuild the model with updated dimensions
        model = rebuild_model(n_users, n_movies, old_model=global_model) # type: ignore
        print(model)
        # Prepare data for training
        X = refined_dataset[['user', 'movie']].values
        y = refined_dataset['rating'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)

        X_train_array = [X_train[:, 0], X_train[:, 1]]
        y_train = y_train / 5

        # Fine-tune the model
        model.fit(x=X_train_array, y=y_train, batch_size=128, epochs=1, shuffle=True)

        # Save the updated model
        model.save("data/movie_recommendation_model.keras")
        global_model = model
        return "Model retrained and saved successfully."
    except Exception as e:
        return f"Failed to retrain model: {str(e)}"
