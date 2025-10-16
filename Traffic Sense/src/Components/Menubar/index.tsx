// Sidebar.tsx
import React from "react";
import "./style.css";
import { RiSpeedUpLine } from "react-icons/ri";
import { NavLink } from "react-router-dom";
import { IoPersonOutline, IoWarningOutline } from "react-icons/io5";
import { FiCreditCard } from "react-icons/fi";
import { HiXMark } from "react-icons/hi2";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
  return (
    <>
      <div className={`sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="logo-sidebar">
            <img src="./logo1.png" alt="Logo" className="logo-img" />
          </div>
          <button className="close-sidebar-btn" onClick={onClose} aria-label="Close menu">
            <HiXMark className="close-icon" />
          </button>
        </div>

        <NavLink to="/speed-detection" className="nav-item" onClick={onClose}>
          <span className="icon-sidebar">
            <RiSpeedUpLine />
          </span>
          <span className="nav-text">Speed Detection</span>
        </NavLink>
        <NavLink to="/inout-count" className="nav-item" onClick={onClose}>
          <span className="icon-sidebar">
            <IoPersonOutline />
          </span>
          <span className="nav-text">In/Out Count</span>
        </NavLink>
        <NavLink to="/number-plate" className="nav-item" onClick={onClose}>
          <span className="icon-sidebar">
            <FiCreditCard />
          </span>
          <span className="nav-text">Number Plate</span>
        </NavLink>
        <NavLink to="/wrong-way" className="nav-item" onClick={onClose}>
          <span className="icon-sidebar">
            <IoWarningOutline />
          </span>
          <span className="nav-text">Wrong-Way</span>
        </NavLink>
      </div>
    </>
  );
};

export default Sidebar;