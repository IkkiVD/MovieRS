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

const getRatingsOfUser = (userId: number) => {
  return fetch(process.env.NEXT_PUBLIC_API_URL + `/ratings/${userId}`, {
    method: "GET",

    headers: {
      "Content-Type": "application/json",
    },
  });
};
export default { giveRating, getRatingsOfUser };
