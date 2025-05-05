import Header from "@components/header";
import RatingForm from "@components/movies/ratingForm";
import Head from "next/head";
import { useRouter } from "next/router";

const RatingsPage: React.FC = () => {
  const router = useRouter();
  const { movieId, title, genres } = router.query;

  return (
    <>
      <Head>
        <title>Rate {title} - MovieRS</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Header />
      <main className="text-center md:mt-24 mx-auto md:w-3/5 lg:w-1/2">
        <RatingForm
          movieId={Number(movieId)}
          title={String(title)}
          genres={String(genres)}
        />
      </main>
    </>
  );
};
export default RatingsPage;
