// Names and payload shapes of the backend ↔ webview protocol. Mirrors
// src-tauri/src/events.rs; keep both in sync when adding an event.

export const EVENTS = {
  statusChanged: "status-changed",
  lockChanged: "lock-changed",
  audioLevel: "audio-level",
  streamingPreview: "streaming-preview",
  transcriptionEmpty: "transcription-empty",
  transcriptionComplete: "transcription-complete",
  operationNotice: "operation-notice",
  historyChanged: "history-changed",
  modelStateChanged: "model-state-changed",
} as const;

export type AppStatusState =
  | "idle"
  | "recording"
  | "transcribing"
  | "formatting"
  | "injecting"
  | "error";

/** Serialized `AppStatus` from Rust: `{ state }` or `{ state: "error", message }`. */
export interface AppStatus {
  state: AppStatusState;
  /** Error class, e.g. "mic" (set only when state is "error"). */
  code?: string;
  message?: string;
}

export const IDLE_STATUS: AppStatus = { state: "idle" };
