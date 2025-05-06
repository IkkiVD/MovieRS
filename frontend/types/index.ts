export type StatusMessage = {
  message: string;
  status: "error" | "success";
};

export type Movie = {
  movieId: number;
  title: string;
  genres: string;
  prediction?: number;
};

export type Rating = {
  userId: number;
  movieId: number;
  title: string;
  genres: string;
  rating: number;
};
