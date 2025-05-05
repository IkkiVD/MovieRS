import { Movie } from "@types";
import { useRouter } from "next/router";

type Props = {
  movies: Movie[];
};

const MoviesOverview: React.FC<Props> = ({ movies }: Props) => {
  const router = useRouter();
  const handleClick = (movie: Movie) => {
    router.push({
      pathname: "/movies/rate",
      query: {
        movieId: movie.movieId,
        title: movie.title,
        genres: movie.genres,
      },
    });
  };

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {movies.map((movie) => (
        <div
          key={movie.movieId}
          className="relative p-4 border rounded shadow-md bg-white hover:shadow-lg transition-shadow cursor-pointer"
          onClick={() => handleClick(movie)}
        >
          <h2 className="text-xl font-bold text-gray-800">{movie.title}</h2>
          <p className="text-gray-600 mt-2 break-words">{movie.genres}</p>
          {movie.prediction && (
            <p>Match: {(movie.prediction * 100).toFixed(1)}%</p>
          )}
        </div>
      ))}
    </div>
  );
};

export default MoviesOverview;
