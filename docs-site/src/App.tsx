import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import GettingStarted from "./pages/GettingStarted";
import Architecture from "./pages/Architecture";
import MlxCore from "./pages/crates/MlxCore";
import MlxOps from "./pages/crates/MlxOps";
import MlxCpu from "./pages/crates/MlxCpu";
import MlxMetal from "./pages/crates/MlxMetal";
import MlxNn from "./pages/crates/MlxNn";
import MlxIo from "./pages/crates/MlxIo";
import MlxSys from "./pages/crates/MlxSys";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/getting-started" element={<GettingStarted />} />
      <Route path="/architecture" element={<Architecture />} />
      <Route path="/crates/mlx-core" element={<MlxCore />} />
      <Route path="/crates/mlx-ops" element={<MlxOps />} />
      <Route path="/crates/mlx-cpu" element={<MlxCpu />} />
      <Route path="/crates/mlx-metal" element={<MlxMetal />} />
      <Route path="/crates/mlx-nn" element={<MlxNn />} />
      <Route path="/crates/mlx-io" element={<MlxIo />} />
      <Route path="/crates/mlx-sys" element={<MlxSys />} />
    </Routes>
  );
}
