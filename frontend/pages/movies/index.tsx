import Header from "@components/header";
import MoviesOverview from "@components/movies/moviesOverview";
import movieService from "@services/movieService";
import ratingService from "@services/ratingService";
import Head from "next/head";
import { useEffect, useState } from "react";
import useSWR from "swr";

const moviesIndex = () => {
  const [filter, setFilter] = useState<string>("");

  const [userId, setUserId] = useState<string | null>(null);

  useEffect(() => {
    const loggedInUser = sessionStorage.getItem("loggedInUser");
    if (loggedInUser != null) {
      setUserId(loggedInUser);
    }
  }, []);

  const fetcher = async () => {
    if (!userId) {
      throw new Error("Please log in first");
    }

    const [moviesResponse, ratingsResponse] = await Promise.all([
      movieService.getMovies(),
      ratingService.getRatingsOfUser(Number(userId)),
    ]);

    if (!moviesResponse.ok) {
      const errorText = await moviesResponse.text();
      throw new Error(`Unable to fetch the movies: ${errorText}`);
    }

    if (!ratingsResponse.ok) {
      const errorText = await ratingsResponse.text();
      throw new Error(`Unable to fetch the ratings: ${errorText}`);
    }

    const movies = await moviesResponse.json();
    const ratings = await ratingsResponse.json();

    return { movies, ratings };
  };

  const { data, isLoading, error } = useSWR(
    userId ? "FetchMoviesAndRatings" : null,
    fetcher
  );

  const filteredMovies = data?.movies
    ? data.movies.filter((movie: { title: string }) =>
        movie.title.toLowerCase().includes(filter.toLowerCase())
      )
    : [];

  return (
    <div>
      <Head>
        <title>Movies - MovieRS</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Header />
      <main className="text-center md:mt-24 mx-auto md:w-3/5 lg:w-1/2">
        {isLoading && <p>Fetching the data...</p>}
        {error && <p className="text-red-600">{error.message}</p>}
        {data && (
          <>
            <label htmlFor="searchInput">Filter:</label>
            <input
              type="text"
              id="searchInput"
              value={filter}
              onChange={(event) => {
                setFilter(event.target.value);
              }}
              className="border border-gray-300 mb-5 text-sm rounded-lg focus:ring-blue-500 focus:border-blue:500 block w-full p-2.5"
            ></input>
            <MoviesOverview movies={filteredMovies} ratings={data.ratings} />
          </>
        )}
      </main>
    </div>
  );
};
export default moviesIndex;
