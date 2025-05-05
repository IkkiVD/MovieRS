const getMovies = () => {
  return fetch(process.env.NEXT_PUBLIC_API_URL + "/movies", {
    method: "GET",

    headers: {
      "Content-Type": "application/json",
    },
  });
};
const getMovie = (id: number) => {
  return fetch(process.env.NEXT_PUBLIC_API_URL + `/movies/${id}`, {
    method: "GET",

    headers: {
      "Content-Type": "application/json",
    },
  });
};
const giveRating = (userId: number, movieId: number, rating: number) => {
  return fetch(
    process.env.NEXT_PUBLIC_API_URL +
      `/movies/rate/${userId}/movie/${movieId}/${rating}`,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },
    }
  );
};

const getRecommendations = (userId: number) => {
  return fetch(process.env.NEXT_PUBLIC_API_URL + `/recommend/${userId}/50/`, {
    method: "GET",

    headers: {
      "Content-Type": "application/json",
    },
  });
};

export default { getMovie, getMovies, giveRating, getRecommendations };
