import { useEffect, useRef, useState } from "react";
import { FaceLandmarker, DrawingUtils } from "@mediapipe/tasks-vision";
import { AttentionEngine } from "../lib/attention";
import { api, TelemetryStream } from "../lib/api";

const ARC_LEN = 264;

export default function LiveCapture() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const engineRef = useRef(null);
  const streamRef = useRef(null); // TelemetryStream
  const camRef = useRef(null); // MediaStream
  const rafRef = useRef(0);
  const lastT = useRef(-1);
  const fpsRef = useRef({ frames: 0, t0: 0, fps: 0 });

  const [phase, setPhase] = useState("idle"); // idle | loading | live | error
  const [label, setLabel] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [err, setErr] = useState("");
  const [out, setOut] = useState(null);

  async function start() {
    setPhase("loading");
    setErr("");
    try {
      const engine = new AttentionEngine();
      await engine.init();
      engineRef.current = engine;

      const cam = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      camRef.current = cam;
      videoRef.current.srcObject = cam;
      await new Promise((r) => (videoRef.current.onloadedmetadata = r));
      await videoRef.current.play();

      const session = await api.createSession(label || "untitled session");
      setSessionId(session.id);
      streamRef.current = new TelemetryStream(session.id);

      setPhase("live");
      rafRef.current = requestAnimationFrame(loop);
    } catch (e) {
      console.error(e);
      setErr(String(e.message || e));
      setPhase("error");
    }
  }

  async function stop() {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.close();
    camRef.current?.getTracks().forEach((t) => t.stop());
    if (sessionId) await api.endSession(sessionId).catch(() => {});
    setPhase("idle");
    setOut(null);
    setSessionId(null);
  }

  function loop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    if (video.currentTime !== lastT.current && video.videoWidth) {
      lastT.current = video.currentTime;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const now = performance.now();

      const sample = engineRef.current.step(video, now);
      streamRef.current?.push(sample);
      draw(canvas, sample);

      const f = fpsRef.current;
      f.frames++;
      if (now - f.t0 > 1000) {
        f.fps = f.frames;
        f.frames = 0;
        f.t0 = now;
      }
      setOut({ ...sample, fps: f.fps, total_blinks: engineRef.current.totalBlinks });
    }
    rafRef.current = requestAnimationFrame(loop);
  }

  function draw(canvas, sample) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const lm = sample.landmarks;
    if (!lm) return;

    const du = new DrawingUtils(ctx);
    du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
      color: "rgba(255,180,84,0.10)",
      lineWidth: 0.5,
    });
    du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#ffb454", lineWidth: 1.2 });
    du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#ffb454", lineWidth: 1.2 });
    du.drawConnectors(lm, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, {
      color: "rgba(230,237,243,0.25)",
      lineWidth: 1,
    });

    // gaze vector from iris midpoint, scaled by offset from eye centerline
    const w = canvas.width, h = canvas.height;
    const lc = lm[468], rc = lm[473];
    const ox = ((lc.x + rc.x) / 2) * w, oy = ((lc.y + rc.y) / 2) * h;
    const mx = ((lm[133].x + lm[362].x) / 2) * w, my = ((lm[133].y + lm[362].y) / 2) * h;
    const dx = (ox - mx) * 18, dy = (oy - my) * 18;
    ctx.strokeStyle = "#3fb68b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    ctx.lineTo(ox + dx, oy + dy);
    ctx.stroke();
    ctx.fillStyle = "#3fb68b";
    ctx.beginPath();
    ctx.arc(ox + dx, oy + dy, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  useEffect(() => () => stop(), []); // cleanup on unmount

  const score = out?.score ?? 0;
  const ang = ((-90 + (180 * score) / 100) * Math.PI) / 180;

  return (
    <div className="layout">
      <div className="panel">
        <h2>Live feed / face mesh</h2>
        <div className="stage">
          <video ref={videoRef} autoPlay playsInline muted />
          <canvas ref={canvasRef} />
          {phase !== "live" && (
            <div className="overlay-msg">
              {phase === "idle" && "Camera is off. Name the session and press start."}
              {phase === "loading" && "Loading landmark model + camera…"}
              {phase === "error" && `Could not start: ${err}`}
            </div>
          )}
        </div>
        {phase !== "live" ? (
          <div className="controls">
            <input
              placeholder="session label (e.g. 'reading sprint 1')"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
            />
            <button onClick={start} disabled={phase === "loading"}>
              Start session
            </button>
          </div>
        ) : (
          <div className="controls">
            <span className="session-tag">session {sessionId} · streaming 5 Hz</span>
            <button className="danger" onClick={stop}>
              End session
            </button>
          </div>
        )}
      </div>

      <div className="stack">
        <div className="panel center">
          <h2>Fused attention score</h2>
          <svg viewBox="0 0 200 110" className="gauge">
            <path d="M 16 100 A 84 84 0 0 1 184 100" fill="none" stroke="var(--grid)" strokeWidth="10" strokeLinecap="round" />
            <path
              d="M 16 100 A 84 84 0 0 1 184 100"
              fill="none" stroke="var(--amber)" strokeWidth="10" strokeLinecap="round"
              strokeDasharray={ARC_LEN}
              strokeDashoffset={ARC_LEN * (1 - score / 100)}
            />
            <line x1="100" y1="100" x2={100 + 70 * Math.sin(ang)} y2={100 - 70 * Math.cos(ang)} stroke="var(--ink)" strokeWidth="2" />
            <circle cx="100" cy="100" r="4" fill="var(--ink)" />
          </svg>
          <div className="score-num">{out ? Math.round(score) : "--"}</div>
          <div className="score-label">smoothed 0–100</div>
        </div>

        <div className="panel">
          <h2>Component signals</h2>
          <Sig name="Gaze on-screen" v={out?.gaze_on} />
          <Sig name="Gaze stability" v={out?.gaze_stab} />
          <Sig name="Head stability" v={out?.head_stab} />
          <Sig name="Motion calm" v={out?.motion_calm} />
          <Sig name="Blink normality" v={out?.blink_norm} />
        </div>

        <div className="panel">
          <h2>Raw readouts</h2>
          <div className="raw">
            <Cell v={out?.total_blinks ?? 0} label="blinks" />
            <Cell v={out?.blink_rate ?? 0} label="blinks/min" />
            <Cell v={out?.ear?.toFixed(3) ?? "--"} label="eye aspect" />
            <Cell v={out?.yaw?.toFixed(1) ?? "--"} label="yaw °" />
            <Cell v={out?.pitch?.toFixed(1) ?? "--"} label="pitch °" />
            <Cell v={out?.fps ?? "--"} label="fps" />
          </div>
        </div>
      </div>
    </div>
  );
}

function Sig({ name, v }) {
  const pct = v != null ? Math.round(v * 100) : 0;
  return (
    <div className="sig">
      <div className="name">{name}</div>
      <div className="bar"><i style={{ width: `${pct}%` }} /></div>
      <div className="val">{v != null ? pct : "--"}</div>
    </div>
  );
}

function Cell({ v, label }) {
  return (
    <div className="cell">
      <b>{v}</b>
      <span>{label}</span>
    </div>
  );
}
