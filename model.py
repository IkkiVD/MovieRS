import torch as pytorch
import pandas as pd


movies = pd.read_csv('./data/movies.csv')
ratings = pd.read_csv('./data/ratings.csv')

print(movies.head)
print(ratings.head)