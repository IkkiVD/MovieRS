import torch
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, ratings_df):
        self.users = torch.tensor(ratings_df['user'].values, dtype=torch.long)
        self.movies = torch.tensor(ratings_df['movie'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]