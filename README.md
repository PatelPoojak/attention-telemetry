# Attention Telemetry — full-stack

Real-time attention telemetry from face landmarks. Vision runs **client-side**
(MediaPipe Face Landmarker, WASM); the backend handles **session persistence,
WebSocket telemetry ingestion (5 Hz), and analytics**. No video ever leaves
the browser — only derived numeric features are stored.

## Architecture

```
┌─ Browser (React + Vite) ─────────────────────────────┐
│  webcam → MediaPipe Face Landmarker (WASM, local)    │
│  → EAR blinks, iris gaze, head pose, motion variance │
│  → weighted fusion + EMA → score                     │
│  → render @30fps │ stream samples @5Hz ──────────────┼──► WS /ws/telemetry/{id}
└──────────────────────────────────────────────────────┘        │
                                                  ┌─ FastAPI ───▼───────────┐
                                                  │ batched SQLite inserts  │
                                                  │ REST: sessions, samples,│
                                                  │ summary + drop events   │
                                                  └─────────────────────────┘
```

## Run

Backend (terminal 1):
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Frontend (terminal 2):
```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 (proxies /api and /ws to :8000)
```

Open http://localhost:5173, allow camera, name a session, hit start.
End the session, then check the Sessions tab for the score timeline,
attention-drop bands, and summary stats.

## Where things live

- `frontend/src/lib/attention.js` — the entire CV core (the file to show in
  a code walkthrough): landmark indices, EAR blink detection with hysteresis,
  iris-offset gaze, head pose from the transform matrix, rolling-window
  stats, weighted fusion, EMA smoothing. All tunables in one `CFG` block.
- `frontend/src/lib/api.js` — REST client + 5 Hz throttled WebSocket stream.
- `backend/main.py` — FastAPI: sessions CRUD, WS ingest with batched writes,
  per-session summary incl. server-side attention-drop event detection.

## Scope honesty

This estimates **attention-related visual signals** (blink rate, gaze offset
and stability, head pose drift, motion variance). It does not detect fatigue,
focus, intelligence, truthfulness, or any mental or medical state.
