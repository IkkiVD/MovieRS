import { StatusMessage } from "@types";
import classNames from "classnames";
import { useRouter } from "next/router";
import { useState } from "react";

const LoginForm: React.FC = () => {
  const router = useRouter();
  const [id, setId] = useState<string>("");
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(
    null
  );
  const handleSubmit = (event: { preventDefault: () => void }) => {
    event.preventDefault();

    if (Number(id) < 0) {
      setStatusMessage({
        message: "Id must be a positive number",
        status: "error",
      });
      return;
    }

    sessionStorage.setItem("loggedInUser", id);
    setStatusMessage({ message: "Succesfully logged in", status: "success" });

    setTimeout(() => {
      router.push("/");
    }, 500);
  };

  return (
    <div className="max-w-sm m-auto">
      <form onSubmit={handleSubmit}>
        <label htmlFor="idInput" className="block mb-2 text-sm font-medium">
          ID {"(new user -> id:611)"}:
        </label>
        <input
          type="text"
          value={id}
          onChange={(event) => setId(event.target.value)}
          id="idInput"
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
        <div className="row mt-2">
          <button
            type="submit"
            className="text-white bg-blue-700 hover:bg-blue-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center"
          >
            Log in
          </button>
        </div>
      </form>
    </div>
  );
};
export default LoginForm;
