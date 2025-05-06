from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import HttpResponse
from django.http import JsonResponse
from .model_tuning import *
import pandas as pd
import numpy as np
import threading
import json


import logging

logging.basicConfig(level=logging.DEBUG)

movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

@csrf_exempt
def recommend_top_n(request, user_id, n=10):
    # Encode the user_id to get recommendations for
    try:
        encoded_user_id = user_enc.transform([user_id])[0]
    except ValueError:
        return HttpResponse(f"User ID {user_id} not found.", status=400)
 
    # Get all the movie ids
    movie_input = np.arange(len(item_enc.classes_))
 
    user_input = np.full_like(movie_input, encoded_user_id) 
 
    predictions = global_model.predict([user_input, movie_input]).flatten() #type: ignore
     
    predictions_dict = {item_enc.inverse_transform([movie_id])[0]: float(predicted_rating)
                        for movie_id, predicted_rating in zip(movie_input.flatten(), predictions)}
 
     # Sort the dictionary on the predicted ratings and take the top n 
    sorted_predictions = sorted(predictions_dict.items(), key=lambda x: x[1], reverse=True)[:n]
     
    top_n_recommendations = []
    for title, predicted_rating in sorted_predictions:
        movie_details = movies_df[movies_df['title'] == title].iloc[0]
        top_n_recommendations.append({
            "movieId": int(movie_details['movieId']),
            "title": title,
            "genres": movie_details['genres'],
            "prediction": predicted_rating
        }) 
    # Return result as JSON
    response = JsonResponse(top_n_recommendations, safe=False)
    response["Access-Control-Allow-Origin"] = "*"
    return response

def get_movies(request):
    movies_json = movies_df.to_dict(orient='records')
    return HttpResponse(json.dumps(movies_json), content_type="application/json")


@csrf_exempt
def give_rating(request, userId: int, movieId: int, rating: int):
    if rating < 0.0 or rating > 5.0:
        return HttpResponse("Rating must be between 0.0 and 5.0.", status=400)

    ratings_file = 'data/ratings.csv'
    try:
        ratings_df = pd.read_csv(ratings_file)
    except FileNotFoundError:
        ratings_df = pd.DataFrame(columns=['userId', 'movieId', 'rating'])

    existing_rating = ratings_df[
        (ratings_df['userId'] == userId) & (ratings_df['movieId'] == movieId)
    ]

    if not existing_rating.empty:
        ratings_df.loc[
            (ratings_df['userId'] == userId) & (ratings_df['movieId'] == movieId),
            'rating'
        ] = rating
    else:
        new_rating = pd.DataFrame({
            'userId': [userId],
            'movieId': [movieId],
            'rating': [rating],
            'timestamp': [0]
        })
        ratings_df = pd.concat([ratings_df, new_rating], ignore_index=True)

    try:
        ratings_df.to_csv(ratings_file, index=False)
    except Exception as e:
        return HttpResponse(f"Failed to save rating: {str(e)}", status=500)

    # Run it in the background, so it doesn't take a couple of seconds to return a 200 status
    def retrain_wrapper():
        try:
            print("Starting retrain_model...")
            retrain_model()
            global global_model
            global_model = keras.saving.load_model('data/movie_recommendation_model.keras')
            print("Finished retrain_model and reloaded the updated model.")
        except Exception as e:
            print(f"Error in retrain_model: {e}")

    threading.Thread(target=retrain_wrapper).start()

    return HttpResponse("Rating saved and model updated successfully.", status=200)


def get_ratings_of_user(request, userId : int):
    rating_df = pd.read_csv("data/ratings.csv")
    movie_df = pd.read_csv("data/movies.csv")
    merged_df = pd.merge(rating_df, movie_df, on="movieId", how="inner")

    filtered_df = merged_df[merged_df["userId"]==userId]

    user_ratings = filtered_df[["userId", "movieId", "title", "genres", "rating"]]
    
    user_ratings_list = user_ratings.to_dict(orient="records")
    
    response = JsonResponse(user_ratings_list, safe=False)
    response["Access-Control-Allow-Origin"] = "*"
    return response