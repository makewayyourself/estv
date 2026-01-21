import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import MarketingDashboard from "./MarketingDashboard";

const App = () => (
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<MarketingDashboard />} />
      <Route path="/dashboard" element={<MarketingDashboard />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  </BrowserRouter>
);

export default App;
