import { Navigate, Outlet } from "react-router-dom";
import { getToken } from "./auth";
const PrivateRoute = () => {
  const token = getToken();
  return token ? <Outlet /> : <Navigate to="/" />;
};

export default PrivateRoute;
