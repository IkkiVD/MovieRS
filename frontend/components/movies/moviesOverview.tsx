import { Movie } from "@types";

type Props = {
  movies: Movie[];
};

const MoviesOverview: React.FC<Props> = ({ movies }: Props) => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {movies.map((movie) => (
        <div
          key={movie.id}
          className="relative p-4 border rounded shadow-md bg-white hover:shadow-lg transition-shadow cursor-pointer"
        >
          <h2 className="text-xl font-bold text-gray-800">{movie.title}</h2>
          <p className="text-gray-600 mt-2 break-words">{movie.genres}</p>
        </div>
      ))}
    </div>
  );
};

export default MoviesOverview;
