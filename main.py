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
print(data_movies)