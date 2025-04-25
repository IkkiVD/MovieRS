export type StatusMessage = {
  message: string;
  status: "error" | "success";
};

export type Movie = {
  id: number;
  title: string;
  genres: string;
};
