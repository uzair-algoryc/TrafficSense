import React, { useState } from "react";
import "./style.css";
import { MdOutlineEmail } from "react-icons/md";
import { handleForgot } from "../../apis/auth";
import { Spinner, useToast } from "@chakra-ui/react";

const ForgotEmail: React.FC = () => {
  const [formData, setFormData] = useState({ email: "" });
  const [isLoading, setIsLoading] = useState(false);
  const toast = useToast();

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.email.trim()) {
      toast({
        title: "Please enter an email address",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      return;
    }
    setIsLoading(true);
    try {
      const response = await handleForgot(formData.email);
      if (response?.success === true) {
        toast({
          title: response.message,
          status: "success",
          duration: 3000,
          isClosable: true,
        });
      } else {
        toast({
          title: response.error || "Failed to send reset link",
          status: "error",
          duration: 3000,
          isClosable: true,
        });
      }
    } catch (error) {
      console.error("Error:", error);
      toast({
        title: "An error occurred while sending reset link",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-form">
        <div className="logo">
          <img src="./logo1.png" alt="logo" className="logo-img" />
        </div>
        <div className="welcome-container">
          <div className="welcome-message">Enter your email</div>
          <div className="welcome-subtitle">
            Enter the email associated with your account and we'll send you
            instructions to reset your password.
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
          <button
            className="btn-email"
            type="submit"
            disabled={!formData.email.trim() || isLoading}
          >
            {isLoading ? <Spinner size="sm" /> : "Submit"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ForgotEmail;
