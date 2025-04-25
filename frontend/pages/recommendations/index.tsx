import Header from "@components/header";
import Head from "next/head";

const recommendationsIndex = () => {
  return (
    <>
      <Head>
        <title>Recommendations - MovieRS</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Header />
      <main className="text-center md:mt-24 mx-auto md:w-3/5 lg:w-1/2">
        <p className="pl-6 text-4xl text-gray-800">Coming soon</p>
      </main>
    </>
  );
};
export default recommendationsIndex;
