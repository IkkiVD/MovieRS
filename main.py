import os
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


data_movies = pd.read_csv("data/movies.csv")
data_ratings = pd.read_csv("data/ratings.csv")

merged_dataset = pd.merge(data_movies,data_ratings, how='inner', on='movieId')

refined_dataset = merged_dataset.groupby(by=['userId','title'], as_index=False).agg({"rating":"mean"})

# Encode the userid and movie title to prepare for DNN
user_enc = LabelEncoder()
refined_dataset['user'] = user_enc.fit_transform(refined_dataset['userId'].to_numpy())
item_enc = LabelEncoder()
refined_dataset['movie'] = item_enc.fit_transform(refined_dataset['title'].to_numpy())
refined_dataset['rating'] = refined_dataset['rating'].values.astype(np.float32)

n_users = refined_dataset['user'].nunique()
n_movies = refined_dataset['movie'].nunique()
min_rating = min(refined_dataset['rating'])
max_rating = max(refined_dataset['rating'])


# Define the x and y data
X = refined_dataset[['user', 'movie']].values
y = refined_dataset['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)

n_factors = 150

X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]

y_train = (y_train - min_rating)/(max_rating - min_rating)
y_test = (y_test - min_rating)/(max_rating - min_rating)

# DNN model

# Input layer for the users
user = keras.layers.Input(shape = (1,))

# Embedding layer for n_factors of users
u = keras.layers.Embedding(n_users, n_factors, embeddings_initializer = 'he_normal', embeddings_regularizer = keras.regularizers.l2(1e-6))(user)
u = keras.layers.Reshape((n_factors,))(u)

# Input layer for the movies
movie = keras.layers.Input(shape = (1,))

# Embedding layer for n_factors of movies
m = keras.layers.Embedding(n_movies, n_factors, embeddings_initializer = 'he_normal', embeddings_regularizer= keras.regularizers.l2(1e-6))(movie)
m = keras.layers.Reshape((n_factors,))(m)

# stacking up both user and movie embeddings
x = keras.layers.Concatenate()([u,m])
x = keras.layers.Dropout(0.05)(x)

# Adding a Dense layer to the architecture
x = keras.layers.Dense(32, kernel_initializer='he_normal')(x)
x = keras.layers.Activation(activation='relu')(x)
x = keras.layers.Dropout(0.05)(x)

x = keras.layers.Dense(16, kernel_initializer='he_normal')(x)
x = keras.layers.Activation(activation='relu')(x)
x = keras.layers.Dropout(0.05)(x)

# Output layer 
x = keras.layers.Dense(1)(x)
x = keras.layers.Activation(activation='linear')(x)

# Define the model
model = keras.models.Model(inputs=[user,movie], outputs=x)

# Compiling the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(model.summary())

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=3, min_lr=0.000001, verbose=1)

history = model.fit(x = X_train_array, y = y_train, batch_size=128, epochs=70, validation_data=(X_test_array, y_test), shuffle=True,callbacks=[reduce_lr])