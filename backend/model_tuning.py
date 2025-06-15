from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
import keras

global_model = keras.saving.load_model('data/movie_recommendation_model.keras')
movies_df = pd.read_csv('data/movies.csv')
user_enc = joblib.load('data/user_enc.pkl')
item_enc = joblib.load('data/item_enc.pkl')

def retrain_model():
    global global_model
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
        # Prepare data for training
        X = refined_dataset[['user', 'movie']].values
        y = refined_dataset['rating'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)

        y_train = y_train / 5
        y_test = y_test / 5

        X_train_array = [X_train[:, 0], X_train[:, 1]]

        # Fine-tune the model
        model.fit(x=X_train_array, y=y_train, batch_size=128, epochs=1, shuffle=True)

        # Save the updated model
        model.save("data/movie_recommendation_model.keras")
        global_model = model
        return "Model retrained and saved successfully."
    except Exception as e:
        return f"Failed to retrain model: {str(e)}"
