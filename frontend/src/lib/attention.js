/**
 * AttentionEngine — the entire computer-vision core, isolated from React.
 *
 * Pipeline per frame:
 *   webcam frame → MediaPipe Face Landmarker (WASM, runs locally)
 *   → geometric feature extraction (EAR, iris offset, head pose, motion)
 *   → rolling-window statistics → weighted fusion → EMA smoothing.
 *
 * Every term in the score traces to an inspectable geometric quantity.
 * There is no learned scoring model and no server-side inference.
 */

import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

/* ---------------- tunables: all in one place, on purpose ---------------- */
export const CFG = {
  EAR_CLOSE: 0.2, // eye aspect ratio below this = eye closing
  EAR_OPEN: 0.25, // must rise above this to re-arm (hysteresis)
  BLINK_WINDOW_MS: 60_000, // blink-rate window
  STAT_WINDOW: 90, // frames (~3 s @ 30 fps) for variance windows
  GAZE_MAX_OFFSET: 0.35, // normalized iris offset treated as fully off-screen
  HEAD_DRIFT_FULL: 12, // deg of rolling std-dev = zero head stability
  MOTION_FULL: 0.012, // normalized nose-displacement std = zero calm
  IDEAL_BLINK_RATE: 15, // blinks/min center of the normal band
  BLINK_BAND: 20, // ± band width to zero score
  SCORE_ALPHA: 0.06, // EMA smoothing factor for the fused score
  WEIGHTS: { gazeOn: 0.3, gazeStab: 0.2, head: 0.2, motion: 0.15, blink: 0.15 },
};

/* ------------- landmark index sets (MediaPipe 478-pt topology) ----------- */
const L_EYE = [33, 160, 158, 133, 153, 144]; // 6-pt contour for EAR
const R_EYE = [362, 385, 387, 263, 373, 380];
const L_IRIS = 468;
const R_IRIS = 473;
const NOSE_TIP = 1;

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const clamp01 = (v) => Math.min(1, Math.max(0, v));

/** Fixed-size ring buffer with mean / std-dev. */
class Rolling {
  constructor(n) {
    this.n = n;
    this.buf = [];
  }
  push(v) {
    this.buf.push(v);
    if (this.buf.length > this.n) this.buf.shift();
  }
  mean() {
    return this.buf.length ? this.buf.reduce((a, b) => a + b, 0) / this.buf.length : 0;
  }
  std() {
    if (this.buf.length < 2) return 0;
    const m = this.mean();
    return Math.sqrt(this.buf.reduce((a, b) => a + (b - m) ** 2, 0) / this.buf.length);
  }
}

/* ----------------------------- features --------------------------------- */

/** EAR = (|p2-p6| + |p3-p5|) / (2|p1-p4|). Open eye ≈ 0.3, closed ≈ 0.05. */
function eyeAspectRatio(lm, idx) {
  const [p1, p2, p3, p4, p5, p6] = idx.map((i) => lm[i]);
  return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4));
}

/** Iris-center offset from the eye-corner midpoint, normalized by eye width.
 *  ~0 = iris centered (roughly screen-directed); larger = looking away. */
function gazeOffset(lm) {
  const per = [
    [L_IRIS, 33, 133],
    [R_IRIS, 362, 263],
  ].map(([iris, a, b]) => {
    const eyeW = dist(lm[a], lm[b]);
    const cx = (lm[a].x + lm[b].x) / 2;
    const cy = (lm[a].y + lm[b].y) / 2;
    return Math.hypot(lm[iris].x - cx, lm[iris].y - cy) / eyeW;
  });
  return (per[0] + per[1]) / 2;
}

/** Yaw/pitch from the 4×4 facial transformation matrix (column-major). */
function headAngles(m) {
  const yaw = (Math.atan2(-m[8], Math.hypot(m[0], m[4])) * 180) / Math.PI;
  const pitch = (Math.atan2(m[9], m[10]) * 180) / Math.PI;
  return { yaw, pitch };
}

/* ------------------------------ engine ----------------------------------- */

export class AttentionEngine {
  constructor() {
    this.landmarker = null;
    this.reset();
  }

  reset() {
    this.blinkArmed = true;
    this.blinkTimes = [];
    this.totalBlinks = 0;
    this.winGaze = new Rolling(CFG.STAT_WINDOW);
    this.winYaw = new Rolling(CFG.STAT_WINDOW);
    this.winPitch = new Rolling(CFG.STAT_WINDOW);
    this.winMotion = new Rolling(CFG.STAT_WINDOW);
    this.prevNose = null;
    this.smoothScore = null;
  }

  async init() {
    const fileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFacialTransformationMatrixes: true,
    });
  }

  /** Process one video frame. Returns a full telemetry sample (or face_seen:0). */
  step(video, now) {
    const result = this.landmarker.detectForVideo(video, now);
    const lm = result.faceLandmarks?.[0];

    if (!lm) {
      // decay score toward 0 while no face is visible
      if (this.smoothScore !== null)
        this.smoothScore += CFG.SCORE_ALPHA * (0 - this.smoothScore);
      return { face_seen: 0, score: this.smoothScore ?? 0, landmarks: null };
    }

    /* blink detection: EAR threshold + hysteresis latch */
    const ear = (eyeAspectRatio(lm, L_EYE) + eyeAspectRatio(lm, R_EYE)) / 2;
    if (this.blinkArmed && ear < CFG.EAR_CLOSE) {
      this.blinkArmed = false;
      this.totalBlinks++;
      this.blinkTimes.push(now);
    } else if (!this.blinkArmed && ear > CFG.EAR_OPEN) {
      this.blinkArmed = true;
    }
    this.blinkTimes = this.blinkTimes.filter((t) => now - t < CFG.BLINK_WINDOW_MS);

    /* gaze */
    const gaze = gazeOffset(lm);
    this.winGaze.push(gaze);

    /* head pose */
    let yaw = 0,
      pitch = 0;
    const mat = result.facialTransformationMatrixes?.[0]?.data;
    if (mat) {
      ({ yaw, pitch } = headAngles(mat));
      this.winYaw.push(yaw);
      this.winPitch.push(pitch);
    }

    /* motion */
    const nose = lm[NOSE_TIP];
    if (this.prevNose) this.winMotion.push(dist(nose, this.prevNose));
    this.prevNose = { x: nose.x, y: nose.y };

    /* ----- fusion: each component mapped to [0,1], 1 = consistent with
       screen-directed attention, then weighted sum + EMA smoothing ----- */
    const w = CFG.WEIGHTS;
    const gazeOn = clamp01(1 - gaze / CFG.GAZE_MAX_OFFSET);
    const gazeStab = clamp01(1 - this.winGaze.std() / 0.08);
    const headDrift = (this.winYaw.std() + this.winPitch.std()) / 2;
    const headStab = clamp01(1 - headDrift / CFG.HEAD_DRIFT_FULL);
    const motionCalm = clamp01(1 - this.winMotion.std() / CFG.MOTION_FULL);
    const rate = this.blinkTimes.length;
    const blinkNorm = clamp01(1 - Math.abs(rate - CFG.IDEAL_BLINK_RATE) / CFG.BLINK_BAND);

    const raw =
      100 *
      (w.gazeOn * gazeOn +
        w.gazeStab * gazeStab +
        w.head * headStab +
        w.motion * motionCalm +
        w.blink * blinkNorm);

    this.smoothScore =
      this.smoothScore === null
        ? raw
        : this.smoothScore + CFG.SCORE_ALPHA * (raw - this.smoothScore);

    return {
      face_seen: 1,
      score: this.smoothScore,
      gaze_on: gazeOn,
      gaze_stab: gazeStab,
      head_stab: headStab,
      motion_calm: motionCalm,
      blink_norm: blinkNorm,
      blink_rate: rate,
      total_blinks: this.totalBlinks,
      ear,
      yaw,
      pitch,
      gaze,
      landmarks: lm,
    };
  }
}
