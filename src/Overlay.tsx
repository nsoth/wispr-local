import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { EVENTS, type AppStatus, type LanguageDetected, type OverlayState } from "./ipc";
import "./styles/overlay.css";

const BAR_COUNT = 28;

/** m:ss for the recording timer. */
export function formatTimer(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

export default function Overlay() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const levelsRef = useRef<number[]>(new Array(BAR_COUNT).fill(0));
  const rafRef = useRef<number | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isLocked, setIsLocked] = useState(false);
  const [overlay, setOverlay] = useState<OverlayState>({
    phase: "recording",
    message: "",
    tone: "",
    language: "",
  });
  const [language, setLanguage] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const startedAtRef = useRef<number | null>(null);

  useEffect(() => {
    const unStatus = listen<AppStatus>(EVENTS.statusChanged, (e) => {
      const recording = e.payload.state === "recording";
      setIsRecording(recording);
      if (!recording) setIsLocked(false);
      if (recording) {
        if (startedAtRef.current === null) startedAtRef.current = Date.now();
      } else if (e.payload.state === "idle" || e.payload.state === "error") {
        startedAtRef.current = null;
        setElapsed(0);
      }
    });

    const unLock = listen<boolean>(EVENTS.lockChanged, (e) => {
      setIsLocked(e.payload);
    });

    const unLevel = listen<number>(EVENTS.audioLevel, (e) => {
      // Shift the ring buffer and push the new sample at the right edge. The
      // bars age by shifting (20 Hz); no per-frame decay, which collapsed all
      // but the newest bars at 60+ fps.
      const arr = levelsRef.current;
      arr.shift();
      // RMS tops out well below 1.0 for speech — stretch to fill the bar.
      arr.push(Math.min(1, e.payload * 3.5));
    });

    const unOverlay = listen<OverlayState>(EVENTS.overlayState, (e) => {
      setOverlay(e.payload);
      if (e.payload.phase === "recording") {
        setLanguage(e.payload.language.toUpperCase());
        if (startedAtRef.current === null) startedAtRef.current = Date.now();
      }
    });

    const unLanguage = listen<LanguageDetected>(EVENTS.languageDetected, (e) => {
      setLanguage(e.payload.language.toUpperCase());
    });

    return () => {
      unStatus.then((fn) => fn());
      unLock.then((fn) => fn());
      unLevel.then((fn) => fn());
      unOverlay.then((fn) => fn());
      unLanguage.then((fn) => fn());
    };
  }, []);

  // Recording timer, 1 Hz.
  useEffect(() => {
    if (!isRecording) return;
    const tick = () => {
      const started = startedAtRef.current ?? Date.now();
      setElapsed((Date.now() - started) / 1000);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [isRecording]);

  // Waveform.
  useEffect(() => {
    if (!isRecording) {
      levelsRef.current.fill(0);
      return;
    }

    const draw = () => {
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          const dpr = window.devicePixelRatio || 1;
          const cssW = canvas.clientWidth;
          const cssH = canvas.clientHeight;
          const pixelW = Math.round(cssW * dpr);
          const pixelH = Math.round(cssH * dpr);
          if (canvas.width !== pixelW || canvas.height !== pixelH) {
            canvas.width = pixelW;
            canvas.height = pixelH;
          }
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, cssW, cssH);

          const arr = levelsRef.current;
          const gap = 2;
          const barW = (cssW - gap * (BAR_COUNT - 1)) / BAR_COUNT;
          const midY = cssH / 2;
          const maxH = cssH * 0.8;

          ctx.fillStyle = "rgba(239, 68, 68, 0.95)";

          for (let i = 0; i < BAR_COUNT; i++) {
            const h = Math.max(2, arr[i] * maxH);
            const x = i * (barW + gap);
            const y = midY - h / 2;
            roundedRect(ctx, x, y, barW, h, Math.min(barW / 2, 2));
            ctx.fill();
          }
        }
      }
      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [isRecording]);

  const handlePinClick = () => {
    invoke("toggle_recording_lock").catch((e) =>
      console.error("toggle_recording_lock failed:", e),
    );
  };

  const phase = isRecording ? "recording" : overlay.phase;
  const classes = [
    "overlay-pill",
    `phase-${phase}`,
    isRecording ? "recording" : "",
    isLocked ? "locked" : "",
    overlay.tone ? `tone-${overlay.tone}` : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={classes} role="status" aria-live="polite">
      <div className="overlay-dot" />
      {phase === "recording" ? (
        <>
          <canvas ref={canvasRef} className="overlay-canvas" />
          {language && <span className="overlay-lang">{language}</span>}
          <span className="overlay-timer">{formatTimer(elapsed)}</span>
          <button
            type="button"
            className="overlay-pin"
            onClick={handlePinClick}
            aria-label={isLocked ? "Stop recording" : "Continue recording hands-free"}
            title={
              isLocked
                ? "Stop recording"
                : "Pin recording (keep going without holding the hotkey)"
            }
          >
            {isLocked ? (
              // Stop icon
              <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
                <rect x="2" y="2" width="8" height="8" rx="1.5" fill="currentColor" />
              </svg>
            ) : (
              // Pin icon
              <svg
                width="13"
                height="13"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M12 17v5" />
                <path d="M9 3h6l-1 7 3 3H7l3-3-1-7z" />
              </svg>
            )}
          </button>
        </>
      ) : (
        <>
          <span className="overlay-text">
            {overlay.message}
            {phase === "processing" ? "…" : ""}
          </span>
          {language && <span className="overlay-lang">{language}</span>}
          {phase === "result" && overlay.tone === "ok" && (
            <svg className="overlay-check" width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M5 12.5l4.5 4.5L19 7.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </>
      )}
    </div>
  );
}

function roundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}
