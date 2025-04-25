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
const likeMovie = (id: number) => {
  return fetch(process.env.NEXT_PUBLIC_API_URL + `/movies/like`, {
    method: "POST",

    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(id),
  });
};

export default { getMovie, getMovies, likeMovie };
