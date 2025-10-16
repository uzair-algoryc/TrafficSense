import { Route, Routes } from "react-router-dom";
import Login from "./Pages/Login";
// import PrivateRoute from "./Components/PrivateFallback";
import SpeedDetection from "./Pages/SpeedDetection";
import Layout from "./Components/layouts";
import InOutCount from "./Pages/InOutCount";
import NumberPlate from "./Pages/NumberPlate";
import WrongWay from "./Pages/WrongWay";
import ForgotEmail from "./Pages/Forgot-Email";
import ForgotPassword from "./Pages/Forgot-Password";

function App (){
  return(
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="/forgot-email" element={<ForgotEmail />} />
      <Route path="/forgot-password/:token" element={<ForgotPassword />} />
      <Route
      //  element={<PrivateRoute />}
      >
        <Route path="/speed-detection" element={
            <Layout>
              <SpeedDetection />
            </Layout>
          } />
      </Route>
      <Route
      //  element={<PrivateRoute />}
      >
        <Route path="/inout-count" element={
            <Layout>
              <InOutCount />
            </Layout>
          } />
      </Route>
      <Route
      //  element={<PrivateRoute />}
      >
        <Route path="/number-plate" element={
            <Layout>
              <NumberPlate />
            </Layout>
          } />
      </Route>
      <Route
      //  element={<PrivateRoute />}
      >
        <Route path="/wrong-way" element={
            <Layout>
              <WrongWay />
            </Layout>
          } />
      </Route>
    </Routes>
  )
}

export default App;