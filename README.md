# MovieRS

MovieRS is a movie recommendation system built using the MovieLens dataset. It allows users to explore movies, rate them, and receive personalized recommendations based on their preferences. The system is powered by a Deep Neural Network (DNN) with embeddings to predict movies you might like.

---

## Features

- **Homepage**: Provides an overview of the application and its functionality.
- **Movies Tab**: Displays all movies from the MovieLens dataset. Users can click on a movie to give it a rating.
- **Recommendations Tab**: Shows 50 personalized movie recommendations based on the movies you have rated.
- **User Login**: Login with a user ID. New users can use the ID `611` to start rating movies and receive recommendations.

---

## How It Works

1. **Login**: Enter your user ID to log in. New users should use the ID `611`.
2. **Rate Movies**: Navigate to the Movies tab and rate a few movies to help the system understand your preferences.
3. **Get Recommendations**: Go to the Recommendations tab to see 50 movies tailored to your taste.

The recommendation engine uses a Deep Neural Network (DNN) with embeddings to analyze user preferences and suggest movies you might enjoy.

---

## Data

The project uses two data folders:

1. **`data/`**:

   - Contains the CSV files that the application uses during runtime.
   - Includes the saved model and encoders for the recommendation system.
   - This folder is updated during runtime as users interact with the application (e.g., adding ratings or retraining the model).

2. **`original_data/`**:
   - Contains the original MovieLens dataset.
   - This folder remains unchanged and serves as the baseline dataset for the application.

---

## Model Details

The recommendation system is powered by a Deep Neural Network (DNN) that uses embeddings to represent users and movies. Below are the details of the model architecture:

1. **Input Layers**:

   - Two input layers: one for users and one for movies.

2. **Embedding Layers**:

   - Each user and movie is represented as a dense vector (embedding) of size `n_factors = 150`.
   - The embeddings are initialized using the `he_normal` initializer and regularized with L2 regularization (`1e-6`).

3. **Concatenation**:

   - The user and movie embeddings are concatenated to form a single feature vector.

4. **Dense Layers**:

   - The concatenated embeddings are passed through two dense layers:
     - First dense layer: 32 units with ReLU activation.
     - Second dense layer: 16 units with ReLU activation.
   - Dropout (`0.05`) is applied after each dense layer to prevent overfitting.

5. **Output Layer**:

   - A single neuron with a linear activation function predicts the rating for a given user-movie pair.

6. **Compilation**:

   - The model is compiled with the Adam optimizer, Mean Squared Error (MSE) as the loss function, and Mean Absolute Error (MAE) as a metric.

7. **Training**:
   - The model is trained on normalized ratings (scaled between 0 and 1).
   - Early stopping is used to prevent overfitting, monitoring validation loss with a patience of 3 epochs.

---

## Installation

1. **Set Up the Backend**:

   - Create a virtual environment:
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source venv/bin/activate
       ```
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Start the backend server:
     ```bash
     python manage.py runserver
     ```

2. **Set Up the Frontend**:
   - Navigate to the frontend directory.
   - Create a `.env` file with the following content:
     ```
     NEXT_PUBLIC_API_URL="http://localhost:8000"
     ```
   - Install the frontend dependencies:
     ```bash
     npm install
     ```
   - Start the frontend development server:
     ```bash
     npm start
     ```

---

## Usage

1. Open the application in your browser.
2. Log in with your user ID (new users should use `611`).
3. Rate a few movies from the Movies tab.
4. Navigate to the Recommendations tab to see your personalized movie suggestions.

---

## Technology Stack

- **Backend**: Django, Keras, TensorFlow
- **Frontend**: React, Next.js
- **Database**: MovieLens 100k dataset
- **Model**: Deep Neural Network (DNN) with embeddings for user and movie features

---

## Notes

- New users must rate a few movies before receiving recommendations.
- The system uses embeddings to learn user preferences and predict movie recommendations.
