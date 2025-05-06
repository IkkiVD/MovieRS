import Link from "next/link";
import { useEffect, useState } from "react";

const Header: React.FC = () => {
  const [loggedInUser, setLoggedInUser] = useState<string | null>(null);

  useEffect(() => {
    const userId = sessionStorage.getItem("loggedInUser");
    if (userId != null) {
      setLoggedInUser(userId);
    }
  }, []);

  const handleClick = () => {
    sessionStorage.removeItem("loggedInUser");
    setLoggedInUser(null);
  };
  return (
    <>
      <header className="p-3 mb-3 border-bottom bg-gradient-to-r from-indigo-400 to-cyan-400 flex flex-col items-center">
        <nav className="items-center flex md:flex-row flex-col">
          <Link
            href="/"
            className=" px-4 text-xl text-white  hover:bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-lg"
          >
            Homepage
          </Link>
          <Link
            href="/movies"
            className=" px-4 text-xl text-white  hover:bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-lg"
          >
            Movies
          </Link>
          <Link
            href="/recommendations"
            className=" px-4 text-xl text-white  hover:bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-lg"
          >
            Recommendations
          </Link>
          {!loggedInUser && (
            <Link
              href="/login"
              className=" px-4 text-xl text-white  hover:bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-lg"
            >
              Login
            </Link>
          )}
          {loggedInUser && (
            <Link
              href="/login"
              onClick={handleClick}
              className=" px-4 text-xl text-white  hover:bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-lg"
            >
              Logout
            </Link>
          )}
        </nav>
      </header>
    </>
  );
};
export default Header;
