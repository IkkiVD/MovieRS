import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        return (user_embeds * movie_embeds).sum(dim=1)