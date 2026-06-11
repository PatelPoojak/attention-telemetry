"""
Attention Telemetry — backend service.

Architecture note (this matters in the demo):
  - All computer vision runs CLIENT-SIDE (MediaPipe WASM in the browser).
    No video frames ever reach this server.
  - The browser streams derived telemetry samples (score, blink rate,
    gaze offset, head pose, motion) over a WebSocket at ~5 Hz.
  - This service persists sessions + samples in SQLite and serves
    per-session analytics for the dashboard.

Run:  uvicorn main:app --reload --port 8000
"""

import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_PATH = Path(__file__).parent / "telemetry.db"


# ---------------------------------------------------------------- database
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                label       TEXT NOT NULL DEFAULT '',
                started_at  TEXT NOT NULL,
                ended_at    TEXT
            );
            CREATE TABLE IF NOT EXISTS samples (
                session_id  TEXT NOT NULL REFERENCES sessions(id),
                t_ms        REAL NOT NULL,   -- ms since session start
                score       REAL NOT NULL,   -- fused 0-100
                gaze_on     REAL NOT NULL,   -- component scores 0-1
                gaze_stab   REAL NOT NULL,
                head_stab   REAL NOT NULL,
                motion_calm REAL NOT NULL,
                blink_norm  REAL NOT NULL,
                blink_rate  REAL NOT NULL,   -- blinks/min (rolling)
                ear         REAL NOT NULL,   -- raw eye aspect ratio
                yaw         REAL NOT NULL,
                pitch       REAL NOT NULL,
                face_seen   INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_samples_session
                ON samples (session_id, t_ms);
            """
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Attention Telemetry API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- models
class SessionCreate(BaseModel):
    label: str = ""


class SessionOut(BaseModel):
    id: str
    label: str
    started_at: str
    ended_at: str | None
    sample_count: int
    avg_score: float | None


# ---------------------------------------------------------------- REST
@app.post("/api/sessions", response_model=SessionOut)
def create_session(body: SessionCreate):
    sid = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    with db() as conn:
        conn.execute(
            "INSERT INTO sessions (id, label, started_at) VALUES (?, ?, ?)",
            (sid, body.label, now),
        )
    return SessionOut(
        id=sid, label=body.label, started_at=now,
        ended_at=None, sample_count=0, avg_score=None,
    )


@app.post("/api/sessions/{sid}/end")
def end_session(sid: str):
    now = datetime.now(timezone.utc).isoformat()
    with db() as conn:
        cur = conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
            (now, sid),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "session not found or already ended")
    return {"ok": True, "ended_at": now}


@app.get("/api/sessions", response_model=list[SessionOut])
def list_sessions():
    with db() as conn:
        rows = conn.execute(
            """
            SELECT s.*, COUNT(m.session_id) AS n, AVG(m.score) AS avg_score
            FROM sessions s LEFT JOIN samples m ON m.session_id = s.id
            GROUP BY s.id ORDER BY s.started_at DESC
            """
        ).fetchall()
    return [
        SessionOut(
            id=r["id"], label=r["label"], started_at=r["started_at"],
            ended_at=r["ended_at"], sample_count=r["n"],
            avg_score=round(r["avg_score"], 1) if r["avg_score"] is not None else None,
        )
        for r in rows
    ]


@app.get("/api/sessions/{sid}/samples")
def get_samples(sid: str, downsample: int = 1):
    """Full time-series for charting. downsample=N keeps every Nth row."""
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM samples WHERE session_id = ? ORDER BY t_ms", (sid,)
        ).fetchall()
    if not rows:
        raise HTTPException(404, "no samples for session")
    step = max(1, downsample)
    return [dict(r) for r in rows[::step]]


@app.get("/api/sessions/{sid}/summary")
def get_summary(sid: str):
    """Aggregate analytics: averages, minima, attention-drop events."""
    with db() as conn:
        agg = conn.execute(
            """
            SELECT COUNT(*) n, AVG(score) avg_score, MIN(score) min_score,
                   MAX(score) max_score, AVG(blink_rate) avg_blink_rate,
                   AVG(gaze_on) avg_gaze_on, AVG(head_stab) avg_head_stab,
                   SUM(1 - face_seen) frames_no_face, MAX(t_ms) duration_ms
            FROM samples WHERE session_id = ?
            """,
            (sid,),
        ).fetchone()
        if agg["n"] == 0:
            raise HTTPException(404, "no samples for session")

        # attention-drop events: contiguous runs where score < 40
        rows = conn.execute(
            "SELECT t_ms, score FROM samples WHERE session_id = ? ORDER BY t_ms",
            (sid,),
        ).fetchall()

    drops, in_drop, start = [], False, 0.0
    for r in rows:
        if r["score"] < 40 and not in_drop:
            in_drop, start = True, r["t_ms"]
        elif r["score"] >= 40 and in_drop:
            in_drop = False
            drops.append({"start_ms": start, "end_ms": r["t_ms"]})
    if in_drop:
        drops.append({"start_ms": start, "end_ms": rows[-1]["t_ms"]})

    return {
        "sample_count": agg["n"],
        "duration_ms": agg["duration_ms"],
        "avg_score": round(agg["avg_score"], 1),
        "min_score": round(agg["min_score"], 1),
        "max_score": round(agg["max_score"], 1),
        "avg_blink_rate": round(agg["avg_blink_rate"], 1),
        "avg_gaze_on": round(agg["avg_gaze_on"], 3),
        "avg_head_stab": round(agg["avg_head_stab"], 3),
        "pct_face_visible": round(100 * (1 - agg["frames_no_face"] / agg["n"]), 1),
        "attention_drops": drops,
    }


# ---------------------------------------------------------------- WebSocket ingest
@app.websocket("/ws/telemetry/{sid}")
async def telemetry_ws(ws: WebSocket, sid: str):
    """
    Browser → server stream. Each message is a JSON telemetry sample
    (already-derived features, never raw video). Batched inserts keep
    SQLite happy at 5-10 Hz per client.
    """
    await ws.accept()
    batch: list[tuple] = []
    try:
        while True:
            raw = await ws.receive_text()
            s = json.loads(raw)
            batch.append((
                sid, s["t_ms"], s["score"], s["gaze_on"], s["gaze_stab"],
                s["head_stab"], s["motion_calm"], s["blink_norm"],
                s["blink_rate"], s["ear"], s["yaw"], s["pitch"],
                int(s.get("face_seen", 1)),
            ))
            if len(batch) >= 10:
                _flush(batch)
                batch = []
    except WebSocketDisconnect:
        pass
    finally:
        if batch:
            _flush(batch)


def _flush(batch: list[tuple]) -> None:
    with db() as conn:
        conn.executemany(
            """INSERT INTO samples
               (session_id, t_ms, score, gaze_on, gaze_stab, head_stab,
                motion_calm, blink_norm, blink_rate, ear, yaw, pitch, face_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            batch,
        )
