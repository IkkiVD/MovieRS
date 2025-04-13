from django.shortcuts import HttpResponse
import pandas as pd
import numpy as np
import joblib
import keras
import json


model = keras.saving.load_model('data/movie_recommendation_model.keras')
user_enc = joblib.load('data/user_enc.pkl')
item_enc = joblib.load('data/item_enc.pkl')
movies_df = pd.read_csv('data/movies.csv')

movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

def recommend_top_n(request, user_id, n=10):
    # Encode the user_id to get recommendations for
    try:
        encoded_user_id = user_enc.transform([user_id])[0]
    except ValueError:
        return HttpResponse(f"User ID {user_id} not found.", status=400)

    # Get all the movie ids
    movie_input = np.arange(len(item_enc.classes_))

    user_input = np.full_like(movie_input, encoded_user_id) 

    predictions = model.predict([user_input, movie_input]).flatten() #type: ignore
    
    predictions_dict = {item_enc.inverse_transform([movie_id])[0]: float(predicted_rating)
                        for movie_id, predicted_rating in zip(movie_input.flatten(), predictions)}

    # Sort the dictionary on the predicted ratings and take the top n 
    sorted_predictions = sorted(predictions_dict.items(), key=lambda x: x[1], reverse=True)[:n]
    
    top_n_predictions = dict(sorted_predictions)

    # Return result as JSON
    return HttpResponse(json.dumps(top_n_predictions), content_type="application/json")