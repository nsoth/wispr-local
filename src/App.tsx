import { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { EVENTS, IDLE_STATUS, type AppStatus } from "./ipc";
import "./styles/global.css";

interface SoundSettings {
  start_sound: string;
  stop_sound: string;
  start_volume: number;
  stop_volume: number;
}

type AiProvider = "none" | "openai" | "claude";

// What the backend shows: never the key itself, only whether one is stored.
interface AiSettingsView {
  provider: AiProvider;
  openai_model: string;
  claude_model: string;
  prompt: string;
  openai_key_set: boolean;
  claude_key_set: boolean;
  key_error: string | null;
}

interface StartupDiagnostics {
  settings_error: string | null;
  settings_read_only: boolean;
  unknown_settings_keys: string[];
  history_error: string | null;
  api_key_error: string | null;
}

type ModelState =
  | { state: "loading" }
  | { state: "ready"; backend: string; file: string; fallback: boolean }
  | { state: "missing" }
  | { state: "failed"; error: string };

interface ModelFileInfo {
  name: string;
  size_bytes: number;
  configured: boolean;
  loaded: boolean;
}

type NoticeKind = "info" | "error";
interface Notice {
  text: string;
  kind: NoticeKind;
}

interface InputDeviceInfo {
  name: string;
  sample_rate: number;
  channels: number;
}

// Mirrors the Rust LanguageMode enum (serde "auto"/"ru"/"en").
type LanguageMode = "auto" | "ru" | "en";

function App() {
  const [status, setStatus] = useState<AppStatus>(IDLE_STATUS);
  const [isLoading, setIsLoading] = useState(true);
  const [notice, setNoticeState] = useState<Notice | null>(null);
  const noticeTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const [diagnostics, setDiagnostics] = useState<string[]>([]);
  const [history, setHistory] = useState<string[]>([]);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [streamingPreview, setStreamingPreview] = useState("");
  const [modelState, setModelState] = useState<ModelState>({ state: "loading" });
  const [modelFiles, setModelFiles] = useState<ModelFileInfo[]>([]);
  const [modelsDir, setModelsDir] = useState("");
  const [hotkey, setHotkey] = useState("Ctrl+Shift+Space");
  const [isCapturingHotkey, setIsCapturingHotkey] = useState(false);
  const [hotkeyError, setHotkeyError] = useState("");
  const [startSound, setStartSound] = useState("");
  const [stopSound, setStopSound] = useState("");
  const [startVolume, setStartVolume] = useState(0.3);
  const [stopVolume, setStopVolume] = useState(0.5);
  // Latest sound values for the debounced save (avoids stale closures).
  const soundRef = useRef({ start: "", stop: "", startVolume: 0.3, stopVolume: 0.5 });
  const soundSaveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const [showSettings, setShowSettings] = useState(false);
  const [autostart, setAutostart] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [language, setLanguage] = useState<LanguageMode>("auto");
  const [inputDevices, setInputDevices] = useState<InputDeviceInfo[]>([]);
  const [inputDevice, setInputDevice] = useState("");
  const [refreshingDevices, setRefreshingDevices] = useState(false);
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const [aiSettings, setAiSettings] = useState<AiSettingsView>({
    provider: "none",
    openai_model: "gpt-4o-mini",
    claude_model: "claude-haiku-4-5-20251001",
    prompt: "",
    openai_key_set: false,
    claude_key_set: false,
    key_error: null,
  });
  // Keys typed but not yet saved; cleared after a successful save.
  const [aiKeyDraft, setAiKeyDraft] = useState({ openai: "", claude: "" });
  const aiRef = useRef({ view: aiSettings, draft: { openai: "", claude: "" } });
  const aiSaveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  // Info notices clear themselves; errors stay until dismissed or replaced.
  const setNotice = (text: string, kind: NoticeKind = "info") => {
    clearTimeout(noticeTimerRef.current);
    if (!text) {
      setNoticeState(null);
      return;
    }
    setNoticeState({ text, kind });
    if (kind === "info") {
      noticeTimerRef.current = setTimeout(() => setNoticeState(null), 5000);
    }
  };
  const setError = (text: string) => setNotice(text, "error");

  useEffect(() => {
    let mounted = true;
    const load = async <T,>(command: string, apply: (value: T) => void) => {
      const value = await invoke<T>(command);
      if (mounted) apply(value);
    };
    // State is registered before the windows exist, so these should never
    // fail; if the IPC bridge is not ready yet (observed on some cold starts),
    // retry with a short backoff before telling the user to restart.
    const RETRY_DELAYS = [300, 1000, 3000];
    const loadAll = (attempt: number) => Promise.allSettled([
      load<ModelState>("get_model_state", setModelState),
      load<string>("get_models_dir", setModelsDir),
      load<string>("get_hotkey", setHotkey),
      load<string[]>("get_history", setHistory),
      load<SoundSettings>("get_sound_settings", (sound) => {
        setStartSound(sound.start_sound);
        setStopSound(sound.stop_sound);
        setStartVolume(sound.start_volume);
        setStopVolume(sound.stop_volume);
        soundRef.current = {
          start: sound.start_sound,
          stop: sound.stop_sound,
          startVolume: sound.start_volume,
          stopVolume: sound.stop_volume,
        };
      }),
      load<AiSettingsView>("get_ai_settings", (ai) => {
        setAiSettings(ai);
        aiRef.current.view = ai;
      }),
      load<StartupDiagnostics>("get_startup_diagnostics", (d) => {
        const problems: string[] = [];
        if (d.settings_error) problems.push(d.settings_error);
        if (d.settings_read_only) {
          problems.push("Settings cannot be saved until Wispr Local is restarted.");
        }
        if (d.unknown_settings_keys.length > 0) {
          problems.push(
            `settings.json has keys this build does not know: ${d.unknown_settings_keys.join(", ")}`,
          );
        }
        if (d.history_error) problems.push(d.history_error);
        if (d.api_key_error) problems.push(`API keys could not be decrypted: ${d.api_key_error}`);
        setDiagnostics(problems);
      }),
      load<boolean>("get_autostart", setAutostart),
      load<boolean>("get_show_overlay", setShowOverlay),
      load<LanguageMode>("get_language", setLanguage),
      load<InputDeviceInfo[]>("get_input_devices", setInputDevices),
      load<string>("get_input_device", setInputDevice),
      load<AppStatus>("get_status", setStatus),
    ]).then((results) => {
      if (!mounted) return;
      const failed = results.filter(
        (result): result is PromiseRejectedResult => result.status === "rejected",
      );
      if (failed.length > 0) {
        const reason = String(failed[0].reason);
        if (attempt < RETRY_DELAYS.length) {
          setTimeout(() => {
            if (mounted) void loadAll(attempt + 1);
          }, RETRY_DELAYS[attempt]);
          return;
        }
        setError("Some settings could not be loaded. Restart Wispr Local if this persists.");
        void invoke("log_frontend_error", {
          command: "initial-load",
          message: `${failed.length} of ${results.length} calls failed: ${reason}`,
        }).catch(() => undefined);
      }
      setIsLoading(false);
    });
    const initialLoad = loadAll(0);

    const unlisten1 = listen<AppStatus>(EVENTS.statusChanged, (event) => {
      setStatus(event.payload);
      if (event.payload.state !== "recording") {
        setStreamingPreview("");
      }
    });

    const unlisten2 = listen<string[]>(EVENTS.historyChanged, (event) => {
      setHistory(event.payload);
    });

    const unlisten3 = listen<string>(EVENTS.streamingPreview, (event) => {
      setStreamingPreview(event.payload);
    });

    const unlisten4 = listen<string>(EVENTS.transcriptionEmpty, (event) => {
      const messages: Record<string, string> = {
        "too-short": "Recording too short — nothing captured",
        "no-speech": "No speech detected — try again",
        error: "Transcription failed — check logs",
      };
      setNotice(messages[event.payload] ?? "Nothing transcribed");
    });

    const unlisten5 = listen<string>(EVENTS.operationNotice, (event) => {
      setNotice(event.payload);
    });

    // The model loads on a background thread; this fires once it finishes (or
    // fails), flipping the footer indicator without a restart.
    const unlisten6 = listen<ModelState>(EVENTS.modelStateChanged, (event) => {
      setModelState(event.payload);
    });

    return () => {
      mounted = false;
      void initialLoad;
      unlisten1.then((fn) => fn());
      unlisten2.then((fn) => fn());
      unlisten3.then((fn) => fn());
      unlisten4.then((fn) => fn());
      unlisten5.then((fn) => fn());
      unlisten6.then((fn) => fn());
      clearTimeout(noticeTimerRef.current);
      clearTimeout(copiedTimerRef.current);
      clearTimeout(soundSaveTimer.current);
      clearTimeout(aiSaveTimer.current);
    };
    // setNotice/setError are stable closures over refs and state setters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const keyCodeToName = (e: KeyboardEvent): string | null => {
    const key = e.key;
    const code = e.code;

    if (["Control", "Shift", "Alt", "Meta"].includes(key)) return null;

    if (code === "Space") return "Space";
    if (code === "Enter") return "Enter";
    if (code === "Tab") return "Tab";
    if (code === "Escape") return "Escape";
    if (code === "Backspace") return "Backspace";
    if (code === "Delete") return "Delete";
    if (code.startsWith("Key")) return code.slice(3);
    if (code.startsWith("Digit")) return code.slice(5);
    if (code.startsWith("F") && /^F\d+$/.test(code)) return code;
    if (code === "ArrowUp") return "Up";
    if (code === "ArrowDown") return "Down";
    if (code === "ArrowLeft") return "Left";
    if (code === "ArrowRight") return "Right";
    if (code === "Minus") return "-";
    if (code === "Equal") return "=";
    if (code === "BracketLeft") return "[";
    if (code === "BracketRight") return "]";
    if (code === "Backslash") return "\\";
    if (code === "Semicolon") return ";";
    if (code === "Quote") return "'";
    if (code === "Comma") return ",";
    if (code === "Period") return ".";
    if (code === "Slash") return "/";
    if (code === "Backquote") return "`";

    return key.length === 1 ? key.toUpperCase() : null;
  };

  const handleHotkeyCapture = useCallback(
    (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();

      const keyName = keyCodeToName(e);
      if (!keyName) return;

      if (keyName === "Escape" && !e.ctrlKey && !e.shiftKey && !e.altKey && !e.metaKey) {
        setIsCapturingHotkey(false);
        return;
      }

      if (!e.ctrlKey && !e.shiftKey && !e.altKey && !e.metaKey) {
        setHotkeyError("Include Ctrl, Shift, Alt, or Win to avoid capturing normal typing");
        return;
      }

      const parts: string[] = [];
      if (e.ctrlKey) parts.push("Ctrl");
      if (e.shiftKey) parts.push("Shift");
      if (e.altKey) parts.push("Alt");
      if (e.metaKey) parts.push("Win");
      parts.push(keyName);

      const newHotkey = parts.join("+");

      setIsCapturingHotkey(false);
      setHotkeyError("");

      invoke("set_hotkey", { hotkey: newHotkey })
        .then(() => setHotkey(newHotkey))
        .catch((err) => setHotkeyError(String(err)));
    },
    []
  );

  useEffect(() => {
    if (isCapturingHotkey) {
      window.addEventListener("keydown", handleHotkeyCapture, true);
      return () => window.removeEventListener("keydown", handleHotkeyCapture, true);
    }
  }, [isCapturingHotkey, handleHotkeyCapture]);

  // AI settings are saved 450 ms after the last edit, and only after an edit:
  // the initial load never writes, so a stored key can never be wiped by a
  // page that has not seen it.
  const flushAiSave = () => {
    const { view, draft } = aiRef.current;
    const update = {
      provider: view.provider,
      openai_model: view.openai_model,
      claude_model: view.claude_model,
      prompt: view.prompt,
      openai_api_key: draft.openai.trim() ? draft.openai.trim() : undefined,
      claude_api_key: draft.claude.trim() ? draft.claude.trim() : undefined,
    };
    invoke<AiSettingsView>("set_ai_settings", { update })
      .then((saved) => {
        aiRef.current = { view: saved, draft: { openai: "", claude: "" } };
        setAiSettings(saved);
        setAiKeyDraft({ openai: "", claude: "" });
      })
      .catch((error) => setError(`Could not save AI settings: ${String(error)}`));
  };

  const scheduleAiSave = () => {
    clearTimeout(aiSaveTimer.current);
    aiSaveTimer.current = setTimeout(flushAiSave, 450);
  };

  const updateAiSettings = (updates: Partial<AiSettingsView>) => {
    const next = { ...aiRef.current.view, ...updates };
    aiRef.current.view = next;
    setAiSettings(next);
    scheduleAiSave();
  };

  const updateKeyDraft = (provider: "openai" | "claude", value: string) => {
    const next = { ...aiRef.current.draft, [provider]: value };
    aiRef.current.draft = next;
    setAiKeyDraft(next);
    scheduleAiSave();
  };

  const removeApiKey = (provider: "openai" | "claude") => {
    clearTimeout(aiSaveTimer.current);
    const view = aiRef.current.view;
    const update = {
      provider: view.provider,
      openai_model: view.openai_model,
      claude_model: view.claude_model,
      prompt: view.prompt,
      openai_api_key: provider === "openai" ? "" : undefined,
      claude_api_key: provider === "claude" ? "" : undefined,
    };
    invoke<AiSettingsView>("set_ai_settings", { update })
      .then((saved) => {
        aiRef.current = { view: saved, draft: { openai: "", claude: "" } };
        setAiSettings(saved);
        setAiKeyDraft({ openai: "", claude: "" });
        setNotice("API key removed");
      })
      .catch((error) => setError(`Could not remove the key: ${String(error)}`));
  };

  // Saves only run from explicit user edits (never from the initial load), so
  // a fresh start can never rewrite settings.json with whatever it read.
  const saveSoundSettings = (
    newStart: string,
    newStop: string,
    newStartVolume: number,
    newStopVolume: number,
  ) => {
    soundRef.current = {
      start: newStart,
      stop: newStop,
      startVolume: newStartVolume,
      stopVolume: newStopVolume,
    };
    return invoke("set_sound_settings", {
      startSound: newStart,
      stopSound: newStop,
      startVolume: newStartVolume,
      stopVolume: newStopVolume,
    }).catch((error) => {
      setError(`Could not save sound settings: ${String(error)}`);
    });
  };

  // Sliders fire continuously; persist 250 ms after the last change.
  const scheduleVolumeSave = (which: "start" | "stop", value: number) => {
    if (which === "start") {
      setStartVolume(value);
      soundRef.current.startVolume = value;
    } else {
      setStopVolume(value);
      soundRef.current.stopVolume = value;
    }
    clearTimeout(soundSaveTimer.current);
    soundSaveTimer.current = setTimeout(() => {
      const s = soundRef.current;
      void saveSoundSettings(s.start, s.stop, s.startVolume, s.stopVolume);
    }, 250);
  };

  const pickSoundFile = async (which: "start" | "stop") => {
    const file = await open({
      multiple: false,
      filters: [{ name: "Audio", extensions: ["wav", "mp3", "ogg", "flac"] }],
    });
    if (typeof file === "string") {
      const path = file;
      const s = soundRef.current;
      if (which === "start") {
        setStartSound(path);
        void saveSoundSettings(path, s.stop, s.startVolume, s.stopVolume);
      } else {
        setStopSound(path);
        void saveSoundSettings(s.start, path, s.startVolume, s.stopVolume);
      }
    }
  };

  const clearSound = (which: "start" | "stop") => {
    const s = soundRef.current;
    if (which === "start") {
      setStartSound("");
      void saveSoundSettings("", s.stop, s.startVolume, s.stopVolume);
    } else {
      setStopSound("");
      void saveSoundSettings(s.start, "", s.startVolume, s.stopVolume);
    }
  };

  const testSound = (which: "start" | "stop", volume: number) => {
    invoke<string>("test_sound", { which, volume })
      .then((device) => setNotice(`Played on ${device}`))
      .catch((error) => setError(`Could not play sound: ${String(error)}`));
  };

  const fileName = (path: string) => {
    if (!path) return "";
    const parts = path.replace(/\\/g, "/").split("/");
    return parts[parts.length - 1];
  };

  const copyHistoryItem = (text: string, index: number) => {
    invoke("copy_text", { text })
      .then(() => {
        setCopiedIndex(index);
        clearTimeout(copiedTimerRef.current);
        copiedTimerRef.current = setTimeout(() => setCopiedIndex(null), 1500);
      })
      .catch((error) => setError(`Copy failed: ${String(error)}`));
  };

  const clearHistory = async () => {
    if (!window.confirm("Clear all saved transcription history?")) return;
    try {
      await invoke("clear_history");
      setHistory([]);
      setNotice("History cleared");
    } catch (error) {
      setError(`Could not clear history: ${String(error)}`);
    }
  };

  const updateToggle = async (
    command: string,
    args: Record<string, boolean>,
    rollback: () => void,
  ) => {
    try {
      await invoke(command, args);
    } catch (error) {
      rollback();
      setError(`Could not save setting: ${String(error)}`);
    }
  };

  const refreshModelFiles = async () => {
    try {
      setModelFiles(await invoke<ModelFileInfo[]>("get_model_files"));
    } catch (error) {
      setError(`Could not list models: ${String(error)}`);
    }
  };

  const reloadModel = () => {
    invoke("reload_model")
      .then(() => setNotice("Reloading the model…"))
      .catch((error) => setError(String(error)));
  };

  const chooseModel = (name: string) => {
    invoke("set_model_file", { name })
      .then(() => {
        setNotice(`Loading ${name}…`);
        void refreshModelFiles();
      })
      .catch((error) => setError(String(error)));
  };

  const modelLabel = (file: string) => file.replace(/^ggml-/, "").replace(/\.bin$/, "");
  const formatSize = (bytes: number) => `${(bytes / 1_000_000_000).toFixed(2)} GB`;
  const modelReady = modelState.state === "ready";
  const modelLoading = modelState.state === "loading";

  const refreshInputDevices = async () => {
    setRefreshingDevices(true);
    try {
      setInputDevices(await invoke<InputDeviceInfo[]>("get_input_devices"));
    } catch (error) {
      setError(`Could not refresh microphones: ${String(error)}`);
    } finally {
      setRefreshingDevices(false);
    }
  };

  const hotkeyParts = hotkey.split("+");
  const isRecording = status.state === "recording";
  const isTranscribing = status.state === "transcribing";
  const isFormatting = status.state === "formatting";
  const isInjecting = status.state === "injecting";
  const isProcessing = isTranscribing || isFormatting || isInjecting;
  const hasError = status.state === "error";
  const errorText = status.message || "Something went wrong";

  return (
    <div className="app">
      <div className="header">
        <div className="logo">W</div>
        <span className="app-name">Wispr Local</span>
        <button
          type="button"
          className="settings-toggle"
          onClick={() => {
            const next = !showSettings;
            setShowSettings(next);
            if (next) void refreshModelFiles();
          }}
          aria-label={showSettings ? "Return to dictation status" : "Open settings"}
          aria-expanded={showSettings}
        >
          {showSettings ? "Back" : "Settings"}
        </button>
      </div>

      {diagnostics.map((problem) => (
        <div className="notice-banner error" role="alert" key={problem}>
          {problem}
        </div>
      ))}

      {notice && (
        <div
          className={`notice-banner ${notice.kind}`}
          role={notice.kind === "error" ? "alert" : "status"}
        >
          <span>{notice.text}</span>
          <button
            type="button"
            className="notice-close"
            onClick={() => setNotice("")}
            aria-label="Dismiss notice"
          >
            ×
          </button>
        </div>
      )}

      {!showSettings ? (
        <>
          <div className="main-section">
            <div
              className={`mic-ring-container${
                isRecording ? " recording" : ""
              }${isProcessing ? " processing" : ""}${hasError ? " error" : ""}`}
              aria-hidden="true"
            >
              <div className="mic-pulse"></div>
              <div className="mic-circle">
                <svg
                  width="28"
                  height="28"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                  <line x1="12" y1="19" x2="12" y2="23" />
                  <line x1="8" y1="23" x2="16" y2="23" />
                </svg>
              </div>
            </div>

            <div className={`status-label${hasError ? " error" : ""}`} role="status" aria-live="polite">
              {isLoading
                ? "Loading..."
                : isRecording
                ? "Listening..."
                : isTranscribing
                ? "Transcribing..."
                : isFormatting
                ? "Formatting..."
                : isInjecting
                ? "Pasting..."
                : hasError
                ? errorText
                : "Ready"}
            </div>

            {isRecording && streamingPreview && (
              <div className="streaming-preview" aria-live="polite">
                <div className="streaming-preview-text">{streamingPreview}</div>
              </div>
            )}

            <div className="hotkey-section">
              {isCapturingHotkey ? (
                <div className="hotkey-capture">
                  <span className="hotkey-capture-text">Press new hotkey...</span>
                  <button
                    type="button"
                    className="hotkey-cancel-btn"
                    onClick={() => setIsCapturingHotkey(false)}
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <>
                  <div className="hotkey-row">
                    {hotkeyParts.map((part, i) => (
                      <span key={i}>
                        {i > 0 && <span className="hotkey-plus">+</span>}
                        <kbd>{part}</kbd>
                      </span>
                    ))}
                    <button
                      type="button"
                      className="hotkey-change-btn"
                      onClick={() => {
                        setIsCapturingHotkey(true);
                        setHotkeyError("");
                      }}
                      title="Change hotkey"
                    >
                      Change
                    </button>
                  </div>
                  <div className="hotkey-desc">Hold to dictate, release to paste</div>
                </>
              )}
              {hotkeyError && (
                <div className="hotkey-error" role="alert">{hotkeyError}</div>
              )}
            </div>
          </div>

          {history.length > 0 && (
            <div className="transcript-card">
              <div className="transcript-heading">
                <div className="transcript-label">History</div>
                <button type="button" className="history-clear-btn" onClick={clearHistory}>
                  Clear
                </button>
              </div>
              <div className="history-list">
                {history.map((item, i) => (
                  <div className="history-item" key={`${i}-${item.slice(0, 24)}`}>
                    <div className="history-text" title={item}>
                      {item}
                    </div>
                    <button
                      type="button"
                      className={`history-copy-btn${copiedIndex === i ? " copied" : ""}`}
                      onClick={() => copyHistoryItem(item, i)}
                      title="Copy to clipboard"
                    >
                      {copiedIndex === i ? "Copied" : "Copy"}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="settings-section">
          <div className="settings-group">
            <div className="settings-group-title">General</div>
            <div className="setting-row">
              <span className="setting-label" id="autostart-label">Start with Windows</span>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  aria-labelledby="autostart-label"
                  checked={autostart}
                  onChange={(e) => {
                    const enabled = e.target.checked;
                    setAutostart(enabled);
                    void updateToggle("set_autostart", { enabled }, () => setAutostart(!enabled));
                  }}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
            <div className="setting-row">
              <span className="setting-label" id="overlay-label">Show floating recording bar</span>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  aria-labelledby="overlay-label"
                  checked={showOverlay}
                  onChange={(e) => {
                    const show = e.target.checked;
                    setShowOverlay(show);
                    void updateToggle("set_show_overlay", { show }, () => setShowOverlay(!show));
                  }}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
            <div className="setting-row">
              <label className="setting-label" htmlFor="language-select">Language</label>
              <select
                id="language-select"
                className="setting-select"
                value={language}
                onChange={(e) => {
                  const lang = e.target.value as LanguageMode;
                  const previous = language;
                  setLanguage(lang);
                  invoke("set_language", { language: lang }).catch((error) => {
                    setLanguage(previous);
                    setError(`Could not save language: ${String(error)}`);
                  });
                }}
              >
                <option value="auto">Auto (Russian / English)</option>
                <option value="ru">Russian</option>
                <option value="en">English</option>
              </select>
            </div>
            <div className="setting-row">
              <label className="setting-label" htmlFor="microphone-select">Microphone</label>
              <div className="device-controls">
                <select
                  id="microphone-select"
                  className="setting-select"
                  value={inputDevice}
                  onChange={(event) => {
                    const next = event.target.value;
                    const previous = inputDevice;
                    setInputDevice(next);
                    invoke("set_input_device", { inputDevice: next }).catch((error) => {
                      setInputDevice(previous);
                      setError(`Could not save microphone: ${String(error)}`);
                    });
                  }}
                >
                  <option value="">System default</option>
                  {inputDevice && !inputDevices.some((device) => device.name === inputDevice) && (
                    <option value={inputDevice}>Unavailable: {inputDevice}</option>
                  )}
                  {inputDevices.map((device) => (
                    <option key={device.name} value={device.name}>
                      {device.name} · {Math.round(device.sample_rate / 1000)} kHz
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="device-refresh-btn"
                  disabled={refreshingDevices}
                  onClick={refreshInputDevices}
                  aria-label="Refresh microphone list"
                  title="Refresh microphone list"
                >
                  {refreshingDevices ? "…" : "↻"}
                </button>
              </div>
            </div>
          </div>

          <div className="settings-group">
            <div className="settings-group-title">Sounds</div>

            <div className="sound-row">
              <span className="sound-label">Start recording</span>
              <div className="sound-controls">
                {startSound ? (
                  <>
                    <span className="sound-file" title={startSound}>
                      {fileName(startSound)}
                    </span>
                    <button type="button" className="sound-btn" onClick={() => clearSound("start")}>
                      Reset
                    </button>
                  </>
                ) : (
                  <span className="sound-file default">Built-in</span>
                )}
                <button type="button" className="sound-btn" onClick={() => pickSoundFile("start")}>
                  Browse
                </button>
                <button
                  type="button"
                  className="sound-btn"
                  onClick={() => testSound("start", startVolume)}
                >
                  Test
                </button>
              </div>
            </div>

            <div className="sound-row">
              <span className="sound-label">Stop recording</span>
              <div className="sound-controls">
                {stopSound ? (
                  <>
                    <span className="sound-file" title={stopSound}>
                      {fileName(stopSound)}
                    </span>
                    <button type="button" className="sound-btn" onClick={() => clearSound("stop")}>
                      Reset
                    </button>
                  </>
                ) : (
                  <span className="sound-file default">Built-in</span>
                )}
                <button type="button" className="sound-btn" onClick={() => pickSoundFile("stop")}>
                  Browse
                </button>
                <button
                  type="button"
                  className="sound-btn"
                  onClick={() => testSound("stop", stopVolume)}
                >
                  Test
                </button>
              </div>
            </div>

            <div className="volume-row">
              <span className="sound-label">Start volume</span>
              <input
                type="range"
                min="0"
                max="100"
                value={Math.round(startVolume * 100)}
                aria-label="Start chime volume"
                aria-valuetext={`${Math.round(startVolume * 100)} percent`}
                onChange={(e) => scheduleVolumeSave("start", Number(e.target.value) / 100)}
                className="volume-slider"
              />
              <span className="volume-value">{Math.round(startVolume * 100)}%</span>
            </div>
            <div className="volume-row">
              <span className="sound-label">Stop volume</span>
              <input
                type="range"
                min="0"
                max="100"
                value={Math.round(stopVolume * 100)}
                aria-label="Stop chime volume"
                aria-valuetext={`${Math.round(stopVolume * 100)} percent`}
                onChange={(e) => scheduleVolumeSave("stop", Number(e.target.value) / 100)}
                className="volume-slider"
              />
              <span className="volume-value">{Math.round(stopVolume * 100)}%</span>
            </div>
            <div className="settings-note">
              Chimes play on the current Windows default output device.
            </div>
          </div>

          <div className="settings-group">
            <div className="settings-group-title">Model</div>
            <div className="setting-row">
              <label className="setting-label" htmlFor="model-select">Whisper model</label>
              <div className="device-controls">
                <select
                  id="model-select"
                  className="setting-select"
                  value={modelFiles.find((m) => m.configured)?.name ?? ""}
                  disabled={modelLoading || isProcessing || isRecording}
                  onChange={(e) => chooseModel(e.target.value)}
                >
                  {modelFiles.length === 0 && <option value="">No models found</option>}
                  {modelFiles.map((m) => (
                    <option key={m.name} value={m.name}>
                      {modelLabel(m.name)} · {formatSize(m.size_bytes)}
                      {m.loaded ? " · loaded" : ""}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="device-refresh-btn"
                  disabled={modelLoading || isProcessing || isRecording}
                  onClick={() => {
                    void refreshModelFiles();
                    reloadModel();
                  }}
                  aria-label="Rescan the models folder and reload"
                  title="Rescan the models folder and reload"
                >
                  ↻
                </button>
              </div>
            </div>
            <div className="settings-note">
              Models folder: <span className="model-path">{modelsDir}</span>
              <button
                type="button"
                className="sound-btn"
                onClick={() =>
                  invoke("open_models_dir").catch((error) =>
                    setError(`Could not open model folder: ${String(error)}`),
                  )
                }
              >
                Open folder
              </button>
            </div>
          </div>

          <div className="settings-group">
            <div className="settings-group-title">AI Formatting</div>

            <div className="setting-row">
              <label className="setting-label" htmlFor="ai-provider">Provider</label>
              <select
                id="ai-provider"
                className="setting-select"
                value={aiSettings.provider}
                onChange={(e) =>
                  updateAiSettings({
                    provider: e.target.value as AiProvider,
                  })
                }
              >
                <option value="none">None (raw text)</option>
                <option value="openai">OpenAI</option>
                <option value="claude">Claude</option>
              </select>
            </div>

            {aiSettings.provider === "openai" && (
              <>
                <div className="setting-row">
                  <label className="setting-label" htmlFor="openai-api-key">API Key</label>
                  <div className="key-controls">
                    <input
                      id="openai-api-key"
                      className="setting-input"
                      type="password"
                      value={aiKeyDraft.openai}
                      onChange={(e) => updateKeyDraft("openai", e.target.value)}
                      placeholder={aiSettings.openai_key_set ? "Key stored · type to replace" : "sk-..."}
                      autoComplete="off"
                      spellCheck={false}
                    />
                    {aiSettings.openai_key_set && (
                      <button
                        type="button"
                        className="sound-btn"
                        onClick={() => removeApiKey("openai")}
                        title="Forget the stored OpenAI key"
                      >
                        Remove
                      </button>
                    )}
                  </div>
                </div>
                <div className="setting-row">
                  <label className="setting-label" htmlFor="openai-model">Model</label>
                  <input
                    id="openai-model"
                    className="setting-input"
                    type="text"
                    value={aiSettings.openai_model}
                    onChange={(e) =>
                      updateAiSettings({ openai_model: e.target.value })
                    }
                    placeholder="gpt-4o-mini"
                    spellCheck={false}
                  />
                </div>
              </>
            )}

            {aiSettings.provider === "claude" && (
              <>
                <div className="setting-row">
                  <label className="setting-label" htmlFor="claude-api-key">API Key</label>
                  <div className="key-controls">
                    <input
                      id="claude-api-key"
                      className="setting-input"
                      type="password"
                      value={aiKeyDraft.claude}
                      onChange={(e) => updateKeyDraft("claude", e.target.value)}
                      placeholder={aiSettings.claude_key_set ? "Key stored · type to replace" : "sk-ant-..."}
                      autoComplete="off"
                      spellCheck={false}
                    />
                    {aiSettings.claude_key_set && (
                      <button
                        type="button"
                        className="sound-btn"
                        onClick={() => removeApiKey("claude")}
                        title="Forget the stored Claude key"
                      >
                        Remove
                      </button>
                    )}
                  </div>
                </div>
                <div className="setting-row">
                  <label className="setting-label" htmlFor="claude-model">Model</label>
                  <input
                    id="claude-model"
                    className="setting-input"
                    type="text"
                    value={aiSettings.claude_model}
                    onChange={(e) =>
                      updateAiSettings({ claude_model: e.target.value })
                    }
                    placeholder="claude-haiku-4-5-20251001"
                    spellCheck={false}
                  />
                </div>
              </>
            )}

            {aiSettings.provider !== "none" && (
              <>
                <div className="setting-row prompt-row">
                  <label className="setting-label" htmlFor="formatting-prompt">Prompt</label>
                  <textarea
                    id="formatting-prompt"
                    className="setting-textarea"
                    value={aiSettings.prompt}
                    onChange={(e) =>
                      updateAiSettings({ prompt: e.target.value })
                    }
                    rows={4}
                    placeholder="Custom formatting instructions..."
                  />
                </div>
                {aiSettings.key_error && (
                  <div className="settings-note error">
                    Stored keys could not be decrypted ({aiSettings.key_error}). Enter the key again.
                  </div>
                )}
                <div className="settings-note">
                  Keys are encrypted for your Windows account and never shown again. Transcripts
                  are sent only when AI formatting is enabled.
                </div>
              </>
            )}
          </div>
        </div>
      )}

      <div className="footer">
        <div
          className={`model-indicator ${
            modelLoading ? "" : modelReady ? "ok" : "err"
          }`}
        >
          <span className="dot" />
          {modelState.state === "loading" && "Loading model…"}
          {modelState.state === "ready" &&
            `Model ready · ${modelState.backend} · ${modelLabel(modelState.file)}${
              modelState.fallback ? " (fallback)" : ""
            }`}
          {modelState.state === "missing" && "No Whisper model found"}
          {modelState.state === "failed" && "Model failed to load"}
        </div>
        {modelState.state === "ready" && modelState.backend !== "CUDA" && (
          <div className="model-help">
            Running on the CPU (slow). Quit and start Wispr Local again to retry CUDA.
          </div>
        )}
        {modelState.state === "missing" && (
          <div className="model-help">
            Download <code>ggml-large-v3-turbo.bin</code> to:
            <span className="model-path">{modelsDir}</span>
            <div className="model-actions">
              <button
                type="button"
                className="model-open-btn"
                disabled={!modelsDir}
                onClick={() =>
                  invoke("open_models_dir").catch((error) =>
                    setError(`Could not open model folder: ${String(error)}`),
                  )
                }
              >
                Open folder
              </button>
              <button type="button" className="model-open-btn" onClick={reloadModel}>
                Rescan
              </button>
            </div>
          </div>
        )}
        {modelState.state === "failed" && (
          <div className="model-help">
            {modelState.error}
            <div className="model-actions">
              <button type="button" className="model-open-btn" onClick={reloadModel}>
                Retry
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
