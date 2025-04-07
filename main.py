import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from dataloader import MovieLensDataset
from model import MatrixFactorization
import torch.nn as nn
import torch


ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings['user'] = user_encoder.fit_transform(ratings['userId'])
ratings['movie'] = movie_encoder.fit_transform(ratings['movieId'])

dataset = MovieLensDataset(ratings)


model = MatrixFactorization(
    num_users=ratings['user'].nunique(),
    num_movies=ratings['movie'].nunique()
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(10):
    model.train()
    total_loss = 0
    for users, movies, ratings_batch in dataloader:
        preds = model(users, movies)
        loss = loss_fn(preds, ratings_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")