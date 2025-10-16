import React, { useState } from "react";
import "./style.css";
import { MdOutlineRemoveRedEye } from "react-icons/md";
import { IoLockClosedOutline } from "react-icons/io5";
import { IoMdEyeOff } from "react-icons/io";
import { useParams } from "react-router-dom";
import { handleReset } from "../../apis/auth";
import { useToast } from "@chakra-ui/react";
import { Spinner } from "@chakra-ui/react";

const ForgotPassword: React.FC = () => {
  const { token } = useParams();
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  const toggleNewPasswordVisibility = () => {
    setShowNewPassword((prev) => !prev);
  };

  const toggleConfirmPasswordVisibility = () => {
    setShowConfirmPassword((prev) => !prev);
  };

  const handleNewPasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setNewPassword(value);
  };

  const handleConfirmPasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setConfirmPassword(value);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    if (newPassword.length < 8) {
      toast({
        title: "Password must be at least 8 characters long",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      setIsLoading(false);
      return;
    }

    if (newPassword !== confirmPassword) {
      toast({
        title: "Passwords do not match",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      setIsLoading(false);
      return;
    }

    try {
      const response = await handleReset(
        token || "",
        newPassword,
        confirmPassword
      );

      if (response?.success) {
        toast({
          title: response.message,
          status: "success",
          duration: 3000,
          isClosable: true,
        });
      } else {
        toast({
          title: response.error || "Failed to reset password",
          status: "error",
          duration: 3000,
          isClosable: true,
        });
      }
    } catch (error) {
      console.error("Error:", error);
      toast({
        title: "An error occurred while resetting password",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const passwordsMatch = newPassword === confirmPassword && newPassword.length > 0;
  const isValid = passwordsMatch && newPassword.length >= 8;

  return (
    <div className="login-container">
      <div className="login-form">
        <div className="logo">
          <img src="./logo1.png" alt="logo" className="logo-img" />
        </div>
        <div className="welcome-container">
          <div className="welcome-message">Forgot Password</div>
          <div className="welcome-subtitle">
            Enter your new password and confirm it.
          </div>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <label>New Password</label>
            <div className="input-wrapper">
              <span className="icon">
                <IoLockClosedOutline />
              </span>
              <input
                type={showNewPassword ? "text" : "password"}
                placeholder="Enter your new password"
                value={newPassword}
                onChange={handleNewPasswordChange}
                required
              />
              <span className="eye" onClick={toggleNewPasswordVisibility} role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && toggleNewPasswordVisibility()}>
                {showNewPassword ? <IoMdEyeOff /> : <MdOutlineRemoveRedEye />}
              </span>
            </div>
          </div>
          <div className="input-group">
            <label>Confirm Password</label>
            <div className="input-wrapper">
              <span className="icon">
                <IoLockClosedOutline />
              </span>
              <input
                type={showConfirmPassword ? "text" : "password"}
                placeholder="Confirm your password"
                value={confirmPassword}
                onChange={handleConfirmPasswordChange}
                required
              />
              <span className="eye" onClick={toggleConfirmPasswordVisibility} role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && toggleConfirmPasswordVisibility()}>
                {showConfirmPassword ? <IoMdEyeOff /> : <MdOutlineRemoveRedEye />}
              </span>
            </div>
          </div>
          <button className="btn-password" type="submit" disabled={!isValid || isLoading}>
            {isLoading ? <Spinner height={5} width={5} /> : "Submit"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ForgotPassword;