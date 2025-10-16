// Layout.tsx
import { type ReactNode, useEffect, useState } from "react";
import "./style.css";
import Sidebar from "../Menubar";
import Header from "../Header";

const Layout = ({ children }: { children: ReactNode }) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const toggleSidebar = () => setIsSidebarOpen((prev) => !prev);
  const closeSidebar = () => setIsSidebarOpen(false);

  useEffect(() => {
    const body = document.body;
    if (isSidebarOpen) {
      body.classList.add("sidebar-open-mobile");
      body.style.overflow = "hidden";
    } else {
      body.classList.remove("sidebar-open-mobile");
      body.style.overflow = "unset";
    }

    return () => {
      body.classList.remove("sidebar-open-mobile");
      body.style.overflow = "unset";
    };
  }, [isSidebarOpen]);

  return (
    <div className="dashboard-mainSection">
      <div className="dashboard-left-section">
        <Sidebar isOpen={isSidebarOpen} onClose={closeSidebar} />
      </div>

      <div className="dashboard-right-section">
        <Header onToggle={toggleSidebar} isOpen={isSidebarOpen} />
        <div className="dashboard-content">{children}</div>
      </div>
      {isSidebarOpen && (
        <div className="sidebar-backdrop" onClick={closeSidebar} />
      )}
    </div>
  );
};

export default Layout;