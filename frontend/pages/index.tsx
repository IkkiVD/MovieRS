import Header from "@components/header";
import Head from "next/head";

const homePage = () => {
  return (
    <>
      <Head>
        <title>HomePage - MovieRS</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Header />
      <main className="text-center md:mt-24 mx-auto md:w-3/5 lg:w-1/2">
        <h1 className="text-4xl font-bold text-gray-800">Welcome to MovieRS</h1>
        <p className="mt-6 text-lg text-gray-700">
          MovieRS is a movie recommendation system built using the MovieLens
          100k dataset. It allows you to explore movies, rate them, and receive
          personalized recommendations.
        </p>
        <p className="mt-4 text-lg text-gray-700">
          Navigate to the <strong>Login</strong> tab and log in with a userId,
          if you are a new user choose id 611. Once you are logged in you can
          navigate to the <strong>Movies</strong> tab to browse the available
          movies and give them a rating out of 5. Based on your ratings, the
          system will recommend other movies tailored to your preferences, which
          you can find in the <strong>Recommendation</strong> tab.
        </p>
        <p className="mt-6 text-lg text-gray-700">
          Start exploring and discover your next favorite movie!
        </p>
      </main>
    </>
  );
};
export default homePage;
