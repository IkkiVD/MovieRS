import Link from "next/link";

const Header: React.FC = () => {
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
          <Link
            href="/login"
            className=" px-4 text-xl text-white  hover:bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-lg"
          >
            Login
          </Link>
        </nav>
      </header>
    </>
  );
};
export default Header;
