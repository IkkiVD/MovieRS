import ratingService from "@services/ratingService";
import { StatusMessage } from "@types";
import classNames from "classnames";
import { useRouter } from "next/router";
import { useEffect, useState } from "react";

type Props = {
  movieId: number;
  title: string;
  genres: string;
  userRating: number | null;
};

const RatingForm: React.FC<Props> = ({
  movieId,
  title,
  genres,
  userRating,
}: Props) => {
  const router = useRouter();
  const [rating, setRating] = useState<string>(
    userRating ? userRating.toString() : ""
  );
  const [userId, setUserId] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(
    null
  );
  useEffect(() => {
    const loggedInUser = sessionStorage.getItem("loggedInUser");
    if (loggedInUser != null) {
      setUserId(loggedInUser);
    }
  }, []);
  const handleSubmit = async (event: { preventDefault: () => void }) => {
    event.preventDefault();

    if (rating === "" || Number(rating) < 0 || Number(rating) > 5) {
      setStatusMessage({
        message: "Rating must be between 0 and 5",
        status: "error",
      });
      return;
    }

    if (userId) {
      const response = await ratingService.giveRating(
        Number(userId),
        movieId,
        Number(rating)
      );
      if (response.ok) {
        setStatusMessage({ message: "Successfully rated", status: "success" });
        setTimeout(() => {
          router.push("/movies");
        }, 1000);
      } else {
        setStatusMessage({ message: "Error processing", status: "error" });
      }
    } else {
      setStatusMessage({
        message: "You have to be logged in to rate a movie",
        status: "error",
      });
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
            value={rating}
            disabled={userRating !== null}
            onChange={(event) => {
              setRating(event.target.value);
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
            disabled={userRating !== null}
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
