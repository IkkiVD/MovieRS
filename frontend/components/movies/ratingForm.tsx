import movieService from "@services/movieService";
import { StatusMessage } from "@types";
import classNames from "classnames";
import { useRouter } from "next/router";
import { useState } from "react";

type Props = {
  movieId: number;
  title: string;
  genres: string;
};

const RatingForm: React.FC<Props> = ({ movieId, title, genres }: Props) => {
  const router = useRouter();
  const [rating, setRating] = useState<number | null>();
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(
    null
  );

  const handleSubmit = async (event: { preventDefault: () => void }) => {
    event.preventDefault();

    if (rating && (rating < 0 || rating > 5)) {
      setStatusMessage({
        message: "Rating must be between 0 and 5",
        status: "error",
      });
      return;
    }

    const userId = sessionStorage.getItem("loggedInUser");

    if (userId) {
      const response = await movieService.giveRating(
        Number(userId),
        movieId,
        rating as number
      );
      if (response.ok) {
        setStatusMessage({ message: "Successfully rated", status: "success" });
        setTimeout(() => {
          router.push("/movies");
        }, 1000);
      } else {
        setStatusMessage({ message: "Error processing", status: "error" });
      }
    }
  };

  return (
    <>
      <h2>Give a rating for the movie</h2>
      <h3>{title}</h3>
      <p>{genres}</p>
      <div className="max-w-sm m-auto">
        <form onSubmit={handleSubmit}>
          <label className="block mb-2 text-sm font-medium">Rating</label>
          <input
            type="number"
            value={rating as number}
            onChange={(event) => {
              setRating(Number(event.target.value));
            }}
            className="border border-gray-300 text-sm rounded-lg focus:ring-blue-500 focus:border-blue:500 block w-full p-2.5"
          ></input>
          {statusMessage && (
            <p
              className={classNames({
                " text-red-800": statusMessage.status === "error",
                "text-green-800": statusMessage.status === "success",
              })}
            >
              {statusMessage.message}
            </p>
          )}
          <button
            type="submit"
            className="text-white bg-blue-700 hover:bg-blue-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center"
          >
            Rate
          </button>
        </form>
      </div>
    </>
  );
};
export default RatingForm;
