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
