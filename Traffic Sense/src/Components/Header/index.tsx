// Header.tsx
import React from "react";
import "./style.css";
import { useLocation } from "react-router-dom";
import { HiMenu } from "react-icons/hi";
import { HiXMark } from "react-icons/hi2";
import { RxPerson } from "react-icons/rx";

const Header: React.FC<{ onToggle: () => void; isOpen: boolean }> = ({
  onToggle,
  isOpen,
}) => {
  const location = useLocation();
  const pathname = location.pathname;

  // Map pathname to dynamic title
  const getTitle = () => {
    switch (pathname) {
      case "/speed-detection":
        return "Speed Detection";
      case "/inout-count":
        return "In Out Count";
      case "/number-plate":
        return "Number Plate Detection";
      case "/wrong-way":
        return "Wrong-Way Detection";
    }
  };

  const title = getTitle();
  return (
    <header className="header">
      <div className="header-content">
        <div className="header-left">
          <button className="hamburger" onClick={onToggle} aria-label="Toggle menu">
            {isOpen ? <HiXMark className="hamburger-icon" /> : <HiMenu className="hamburger-icon" />}
          </button>
          <div className="header-title-container">
            <div className="header-title">{title}</div>
            <div className="user-tagline">AI-powered analytics</div>
          </div>
        </div>
        <div className="user-section">
          <div className="user-info">
            <div className="user-details">
              <div className="user-name">Admin User</div>
              <div className="user-email">
                <span className="email-text">admin@trafficsense.com</span>
              </div>
            </div>
            <div className="user-avatar">
              <span className="avatar-icon"><RxPerson color="white"/></span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;