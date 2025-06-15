import os
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom", name="rmse")
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

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

n_factors = 250

X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]
 
y_train = (y_train - min_rating)/(max_rating - min_rating)
y_test = (y_test - min_rating)/(max_rating - min_rating)

# DNN model

# Input layer for the users
user = keras.layers.Input(shape = (1,))

# Embedding layer for n_factors of users
u = keras.layers.Embedding(n_users, n_factors)(user)
u = keras.layers.Reshape((n_factors,))(u)

# Input layer for the movies
movie = keras.layers.Input(shape = (1,))

# Embedding layer for n_factors of movies
m = keras.layers.Embedding(n_movies, n_factors)(movie)
m = keras.layers.Reshape((n_factors,))(m)


user_vec = keras.layers.Flatten()(u)
movie_vec = keras.layers.Flatten()(m)

# dot product to find similarities
x = keras.layers.Dot(axes=1)([user_vec,movie_vec])
x = keras.layers.Dropout(0.05)(x)


x = keras.layers.Dense(32, kernel_initializer='he_normal')(x)
x = keras.layers.Activation(activation='relu')(x)
x = keras.layers.Dropout(0.05)(x)

x = keras.layers.Dense(16, kernel_initializer='he_normal')(x)
x = keras.layers.Activation(activation='relu')(x)
x = keras.layers.Dropout(0.05)(x)

# Output layer 
x = keras.layers.Dense(1)(x)
x = keras.layers.Activation(activation='sigmoid')(x)

# Define the model
model = keras.models.Model(inputs=[user,movie], outputs=x)

# Compiling the model
model.compile(optimizer='adam', loss=rmse, metrics=['mae'])
print(model.summary())

# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)

history = model.fit(x = X_train_array, y = y_train, batch_size=128, epochs=10, validation_data=(X_test_array, y_test), shuffle=True)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss History')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png') 
plt.close()

model.save("data/movie_recommendation_model.keras")