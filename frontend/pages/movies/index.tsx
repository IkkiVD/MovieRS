import Header from "@components/header";
import MoviesOverview from "@components/movies/moviesOverview";
import movieService from "@services/movieService";
import { StatusMessage } from "@types";
import Head from "next/head";
import { useState } from "react";
import useSWR from "swr";

const moviesIndex = () => {
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(
    null
  );

  const fetcher = async () => {
    const response = await movieService.getMovies();
    if (!response.ok) {
      setStatusMessage({
        message: "Unable to fetch the movies",
        status: "error",
      });
    }
    return response.json();
  };

  const { data, isLoading, error } = useSWR("FetchMovies", fetcher);

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
        {data && <MoviesOverview movies={data} />}
      </main>
    </div>
  );
};
export default moviesIndex;
