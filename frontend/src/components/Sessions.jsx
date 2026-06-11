import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis,
  Tooltip, CartesianGrid, ReferenceArea,
} from "recharts";
import { api } from "../lib/api";

const fmtTime = (ms) => `${(ms / 1000).toFixed(0)}s`;

export function SessionList() {
  const [sessions, setSessions] = useState(null);
  useEffect(() => {
    api.listSessions().then(setSessions).catch(() => setSessions([]));
  }, []);

  if (sessions === null) return <div className="panel">Loading sessions…</div>;
  if (sessions.length === 0)
    return (
      <div className="panel">
        <h2>Sessions</h2>
        <p className="empty">No sessions recorded yet. Run a live capture first — every session streams here automatically.</p>
      </div>
    );

  return (
    <div className="panel">
      <h2>Sessions</h2>
      <table className="sessions">
        <thead>
          <tr><th>id</th><th>label</th><th>started</th><th>samples</th><th>avg score</th></tr>
        </thead>
        <tbody>
          {sessions.map((s) => (
            <tr key={s.id}>
              <td><Link to={`/sessions/${s.id}`}>{s.id}</Link></td>
              <td>{s.label || "—"}</td>
              <td>{new Date(s.started_at).toLocaleString()}</td>
              <td>{s.sample_count}</td>
              <td>{s.avg_score ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function SessionDetail() {
  const { id } = useParams();
  const [samples, setSamples] = useState(null);
  const [summary, setSummary] = useState(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    Promise.all([api.getSamples(id), api.getSummary(id)])
      .then(([sa, su]) => { setSamples(sa); setSummary(su); })
      .catch((e) => setErr(String(e.message || e)));
  }, [id]);

  if (err) return <div className="panel">Could not load session: {err}</div>;
  if (!samples) return <div className="panel">Loading session {id}…</div>;

  return (
    <div className="stack">
      <div className="panel">
        <h2>Session {id} — score timeline</h2>
        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={samples} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
            <defs>
              <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ffb454" stopOpacity={0.5} />
                <stop offset="100%" stopColor="#ffb454" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#1d242d" vertical={false} />
            <XAxis dataKey="t_ms" tickFormatter={fmtTime} stroke="#8b949e" fontSize={10} />
            <YAxis domain={[0, 100]} stroke="#8b949e" fontSize={10} />
            <Tooltip
              labelFormatter={(v) => `t = ${fmtTime(v)}`}
              formatter={(v) => [Number(v).toFixed(1), "score"]}
              contentStyle={{ background: "#161b22", border: "1px solid #232a33", fontSize: 11 }}
            />
            {summary?.attention_drops.map((d, i) => (
              <ReferenceArea key={i} x1={d.start_ms} x2={d.end_ms} fill="#e5534b" fillOpacity={0.12} />
            ))}
            <Area type="monotone" dataKey="score" stroke="#ffb454" strokeWidth={1.5} fill="url(#g)" dot={false} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
        <p className="hint">Red bands = attention-drop events (score &lt; 40), computed server-side.</p>
      </div>

      {summary && (
        <div className="panel">
          <h2>Summary analytics</h2>
          <div className="raw wide">
            <Cell v={(summary.duration_ms / 1000).toFixed(0) + "s"} label="duration" />
            <Cell v={summary.sample_count} label="samples" />
            <Cell v={summary.avg_score} label="avg score" />
            <Cell v={summary.min_score} label="min score" />
            <Cell v={summary.max_score} label="max score" />
            <Cell v={summary.avg_blink_rate} label="avg blinks/min" />
            <Cell v={summary.pct_face_visible + "%"} label="face visible" />
            <Cell v={summary.attention_drops.length} label="drop events" />
          </div>
        </div>
      )}

      <Link className="back" to="/sessions">← all sessions</Link>
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
