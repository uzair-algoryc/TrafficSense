import React, { useState } from "react";
import "./style.css";
import { MdOutlineEmail, MdOutlineRemoveRedEye } from "react-icons/md";
import { IoLockClosedOutline } from "react-icons/io5";
import { IoMdEyeOff } from "react-icons/io";
import { NavLink } from "react-router-dom";
import { handleLogin } from "../../apis/auth";
import { useToast, Spinner } from "@chakra-ui/react";
import { useNavigate } from "react-router-dom";

const Login: React.FC = () => {
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({ email: "", password: "" });
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();
  const navigate = useNavigate();

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

const handleAuth = async () => {
  setIsLoading(true);
  const staticEmail = "admin@trafficsense.com";
  const staticPassword = "123456";

  try {
    // ✅ Step 1: Static login check first
    if (
      formData.email === staticEmail &&
      formData.password === staticPassword
    ) {
      localStorage.setItem("login_auth_token", "static_demo_token");
      toast({
        title: "Login successful",
        status: "success",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
      navigate("/speed-detection");
      return; // stop here, don't call API
    }

    // ✅ Step 2: Fallback to real API login if not static
    const response = await handleLogin(formData.email, formData.password);

    if (response.success) {
      const token = response.token;
      localStorage.setItem("login_auth_token", token);
      toast({
        title: response.message,
        status: "success",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
      navigate("/speed-detection");
    } else {
      toast({
        title: response.message,
        status: "error",
        duration: 3000,
        isClosable: true,
        position: "top",
      });
    }
  } catch (error) {
    console.error(error);
    toast({
      title: "Something went wrong",
      status: "error",
      duration: 3000,
      isClosable: true,
      position: "top",
    });
  } finally {
    setIsLoading(false);
  }
};


  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleAuth();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  return (
    <div className="login-container">
      <div className="login-form">
        <div className="logo">
          <img src="./logo1.png" alt="logo" className="logo-img" />
        </div>
        <div className="welcome-container">
          <div className="welcome-message">Welcome Back</div>
          <div className="welcome-subtitle">
            Sign in to access your Traffic Sense dashboard
          </div>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <label htmlFor="email">Email Address</label>
            <div className="input-wrapper">
              <span className="icon">
                <MdOutlineEmail />
              </span>
              <input
                id="email"
                type="email"
                name="email"
                placeholder="admin@trafficsense.com"
                value={formData.email}
                onChange={handleInputChange}
                required
              />
            </div>
          </div>
          <div className="input-group">
            <label htmlFor="password">Password</label>
            <div className="input-wrapper">
              <span className="icon">
                <IoLockClosedOutline />
              </span>
              <input
                id="password"
                type={showPassword ? "text" : "password"}
                name="password"
                placeholder="Enter your password"
                value={formData.password}
                onChange={handleInputChange}
                required
              />
              <span className="eye" onClick={togglePasswordVisibility}>
                {showPassword ? <IoMdEyeOff /> : <MdOutlineRemoveRedEye />}
              </span>
            </div>
          </div>
          <div className="forgot-container">
            <div className="checkbox-group">
              <input type="checkbox" id="remember" />
              <label htmlFor="remember">Remember me</label>
            </div>
            <div className="forgot-password">
              <NavLink to="/forgot-email" className="forgot">
                Forgot password?
              </NavLink>
            </div>
          </div>
          <button className="btn-login" type="submit" disabled={isLoading}>
            {isLoading ? <Spinner size="sm" /> : "Sign In"}
          </button>
        </form>
        <p className="no-account">
          Don't have an account? <a href="#">Contact Sales</a>
        </p>
      </div>
    </div>
  );
};

export default Login;