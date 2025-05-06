import Header from "@components/header";
import MoviesOverview from "@components/movies/moviesOverview";
import movieService from "@services/movieService";
import ratingService from "@services/ratingService";
import { Movie, StatusMessage } from "@types";
import Head from "next/head";
import { useState } from "react";
import useSWR from "swr";

const recommendationsIndex = () => {
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(
    null
  );

  const fetcher = async () => {
    const userId = sessionStorage.getItem("loggedInUser");

    if (userId) {
      const ratingsResponse = await ratingService.getRatingsOfUser(
        Number(userId)
      );
      if (!ratingsResponse.ok) {
        setStatusMessage({
          message: "Unable to fetch the ratings of the user",
          status: "error",
        });
        return;
      }

      const ratings = await ratingsResponse.json();

      const response = await movieService.getRecommendations(Number(userId));
      if (!response.ok) {
        setStatusMessage({
          message: "Unable to fetch the movies",
          status: "error",
        });
      }
      return await response.json();
    } else {
      setStatusMessage({ message: "Please log in first...", status: "error" });
    }
  };

  const { data, isLoading, error } = useSWR("fetchRecommendations", fetcher);

  return (
    <>
      <Head>
        <title>Recommendations - MovieRS</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Header />
      <main className="text-center md:mt-24 mx-auto md:w-3/5 lg:w-1/2">
        {isLoading && <p>Fetching the data...</p>}
        {error && <p className="text-red-600">{error.message}</p>}
        {data && <MoviesOverview movies={data} />}{" "}
      </main>
    </>
  );
};
export default recommendationsIndex;
