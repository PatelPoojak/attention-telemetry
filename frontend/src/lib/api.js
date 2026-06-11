/** Thin client for the FastAPI backend. Vite proxies /api and /ws in dev. */

const json = (r) => {
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
};

export const api = {
  createSession: (label) =>
    fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label }),
    }).then(json),
  endSession: (id) => fetch(`/api/sessions/${id}/end`, { method: "POST" }).then(json),
  listSessions: () => fetch("/api/sessions").then(json),
  getSamples: (id, downsample = 1) =>
    fetch(`/api/sessions/${id}/samples?downsample=${downsample}`).then(json),
  getSummary: (id) => fetch(`/api/sessions/${id}/summary`).then(json),
};

/**
 * Streams telemetry samples to the backend over WebSocket, throttled to
 * SEND_HZ. The render loop runs at ~30fps; persisting every frame is
 * pointless, so we sample the stream at 5 Hz.
 */
export class TelemetryStream {
  static SEND_HZ = 5;

  constructor(sessionId) {
    this.t0 = performance.now();
    this.lastSend = 0;
    this.ws = new WebSocket(
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/telemetry/${sessionId}`
    );
  }

  push(sample) {
    const now = performance.now();
    if (now - this.lastSend < 1000 / TelemetryStream.SEND_HZ) return;
    if (this.ws.readyState !== WebSocket.OPEN) return;
    this.lastSend = now;
    this.ws.send(
      JSON.stringify({
        t_ms: now - this.t0,
        score: sample.score ?? 0,
        gaze_on: sample.gaze_on ?? 0,
        gaze_stab: sample.gaze_stab ?? 0,
        head_stab: sample.head_stab ?? 0,
        motion_calm: sample.motion_calm ?? 0,
        blink_norm: sample.blink_norm ?? 0,
        blink_rate: sample.blink_rate ?? 0,
        ear: sample.ear ?? 0,
        yaw: sample.yaw ?? 0,
        pitch: sample.pitch ?? 0,
        face_seen: sample.face_seen,
      })
    );
  }

  close() {
    if (this.ws.readyState === WebSocket.OPEN) this.ws.close();
  }
}
