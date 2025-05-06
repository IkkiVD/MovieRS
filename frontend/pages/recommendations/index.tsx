import Header from "@components/header";
import MoviesOverview from "@components/movies/moviesOverview";
import movieService from "@services/movieService";
import ratingService from "@services/ratingService";
import { Rating } from "@types";
import Head from "next/head";
import useSWR from "swr";

const recommendationsIndex = () => {
  const fetcher = async () => {
    const userId = sessionStorage.getItem("loggedInUser");

    if (userId) {
      const ratingsResponse = await ratingService.getRatingsOfUser(
        Number(userId)
      );
      if (!ratingsResponse.ok) {
        const errorText = await ratingsResponse.text();
        throw new Error(
          `Unable to fetch the ratings of user ${userId}: ${errorText}`
        );
        return;
      }

      const ratings: Rating[] = await ratingsResponse.json();
      console.log(ratings);
      if (ratings.length >= 1) {
        const response = await movieService.getRecommendations(Number(userId));
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Unable to fetch the recommendations: ${errorText}`);
        }
        return await response.json();
      } else {
        throw new Error("Please rate a couple of movies...");
      }
    } else {
      throw new Error("Please log in first");
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
        {data && <MoviesOverview movies={data} ratings={[]} />}
      </main>
    </>
  );
};
export default recommendationsIndex;
