import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import "./styles/overlay.css";

const BAR_COUNT = 28;
const DECAY = 0.82;

export default function Overlay() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const levelsRef = useRef<number[]>(new Array(BAR_COUNT).fill(0));
  const rafRef = useRef<number | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isLocked, setIsLocked] = useState(false);

  useEffect(() => {
    const unStatus = listen<string>("status-changed", (e) => {
      const recording = e.payload === "Recording";
      setIsRecording(recording);
      if (!recording) setIsLocked(false);
    });

    const unLock = listen<boolean>("lock-changed", (e) => {
      setIsLocked(e.payload);
    });

    const unLevel = listen<number>("audio-level", (e) => {
      // Shift the ring buffer and push the new sample at the right edge.
      const arr = levelsRef.current;
      arr.shift();
      // RMS tops out well below 1.0 for speech — stretch to fill the bar.
      const normalized = Math.min(1, e.payload * 3.5);
      arr.push(normalized);
    });

    return () => {
      unStatus.then((fn) => fn());
      unLock.then((fn) => fn());
      unLevel.then((fn) => fn());
    };
  }, []);

  useEffect(() => {
    const draw = () => {
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          const dpr = window.devicePixelRatio || 1;
          const cssW = canvas.clientWidth;
          const cssH = canvas.clientHeight;
          if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
            canvas.width = cssW * dpr;
            canvas.height = cssH * dpr;
          }
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, cssW, cssH);

          const arr = levelsRef.current;
          const gap = 2;
          const barW = (cssW - gap * (BAR_COUNT - 1)) / BAR_COUNT;
          const midY = cssH / 2;
          const maxH = cssH * 0.8;

          ctx.fillStyle = isRecording
            ? "rgba(239, 68, 68, 0.95)"
            : "rgba(148, 163, 184, 0.7)";

          for (let i = 0; i < BAR_COUNT; i++) {
            const v = arr[i];
            // Decay idle bars so the waveform fades when silent.
            if (!isRecording) arr[i] = v * DECAY;
            const h = Math.max(2, v * maxH);
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

  return (
    <div
      className={`overlay-pill${isRecording ? " recording" : ""}${
        isLocked ? " locked" : ""
      }`}
    >
      <div className="overlay-dot" />
      <canvas ref={canvasRef} className="overlay-canvas" />
      {isRecording && (
        <button
          className="overlay-pin"
          onClick={handlePinClick}
          title={
            isLocked
              ? "Stop recording"
              : "Pin recording (keep going without holding the hotkey)"
          }
        >
          {isLocked ? (
            // Stop icon
            <svg width="12" height="12" viewBox="0 0 12 12">
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
            >
              <path d="M12 17v5" />
              <path d="M9 3h6l-1 7 3 3H7l3-3-1-7z" />
            </svg>
          )}
        </button>
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
