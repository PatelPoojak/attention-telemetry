import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import LiveCapture from "./components/LiveCapture";
import { SessionList, SessionDetail } from "./components/Sessions";

export default function App() {
  return (
    <BrowserRouter>
      <header>
        <h1>Attention Telemetry</h1>
        <span className="sub">client-side CV · server-side analytics</span>
        <nav>
          <NavLink to="/" end>Live</NavLink>
          <NavLink to="/sessions">Sessions</NavLink>
        </nav>
      </header>

      <main>
        <Routes>
          <Route path="/" element={<LiveCapture />} />
          <Route path="/sessions" element={<SessionList />} />
          <Route path="/sessions/:id" element={<SessionDetail />} />
        </Routes>
      </main>

      <footer>
        This system estimates attention-related visual signals from observable face
        landmarks (blink rate, gaze offset and stability, head pose drift, motion
        variance). It does not measure fatigue, focus, intelligence, or any mental or
        medical state. All vision processing runs locally in the browser; only derived
        numeric telemetry is stored.
      </footer>
    </BrowserRouter>
  );
}
