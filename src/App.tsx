import { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import "./styles/global.css";

interface SoundSettings {
  start_sound: string;
  stop_sound: string;
  sound_volume: number;
}

interface AiSettings {
  provider: "none" | "openai" | "claude";
  api_key: string;
  openai_model: string;
  claude_model: string;
  prompt: string;
}

interface InputDeviceInfo {
  name: string;
  sample_rate: number;
  channels: number;
}

// Mirrors the Rust LanguageMode enum (serde "auto"/"ru"/"en").
type LanguageMode = "auto" | "ru" | "en";

function App() {
  const [status, setStatus] = useState("Idle");
  const [isLoading, setIsLoading] = useState(true);
  const [notice, setNotice] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [streamingPreview, setStreamingPreview] = useState("");
  const [modelLoaded, setModelLoaded] = useState(false);
  // The model loads asynchronously on the backend; until we get a definitive
  // result (initial query returning true, or the model-state-changed event) we
  // show "Checking model..." instead of the false "Model not loaded" help.
  const [modelReported, setModelReported] = useState(false);
  const [computeBackend, setComputeBackend] = useState("");
  const [modelsDir, setModelsDir] = useState("");
  const [hotkey, setHotkey] = useState("Ctrl+Shift+Space");
  const [isCapturingHotkey, setIsCapturingHotkey] = useState(false);
  const [hotkeyError, setHotkeyError] = useState("");
  const [startSound, setStartSound] = useState("");
  const [stopSound, setStopSound] = useState("");
  const [soundVolume, setSoundVolume] = useState(0.5);
  const [showSettings, setShowSettings] = useState(false);
  const [autostart, setAutostart] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [language, setLanguage] = useState<LanguageMode>("auto");
  const [inputDevices, setInputDevices] = useState<InputDeviceInfo[]>([]);
  const [inputDevice, setInputDevice] = useState("");
  const [refreshingDevices, setRefreshingDevices] = useState(false);
  const [aiSettingsLoaded, setAiSettingsLoaded] = useState(false);
  const [soundSettingsLoaded, setSoundSettingsLoaded] = useState(false);
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const [aiSettings, setAiSettings] = useState<AiSettings>({
    provider: "none",
    api_key: "",
    openai_model: "gpt-4o-mini",
    claude_model: "claude-haiku-4-5-20251001",
    prompt: "",
  });

  useEffect(() => {
    let mounted = true;
    const load = async <T,>(command: string, apply: (value: T) => void) => {
      const value = await invoke<T>(command);
      if (mounted) apply(value);
    };
    const initialLoad = Promise.allSettled([
      load<boolean>("is_model_loaded", (value) => {
        setModelLoaded(value);
        // A positive result is definitive; a negative one may just mean the
        // async load hasn't finished, so wait for model-state-changed.
        if (value) setModelReported(true);
      }),
      load<string>("get_compute_backend", setComputeBackend),
      load<string>("get_models_dir", setModelsDir),
      load<string>("get_hotkey", setHotkey),
      load<string[]>("get_history", setHistory),
      load<SoundSettings>("get_sound_settings", (sound) => {
        setStartSound(sound.start_sound);
        setStopSound(sound.stop_sound);
        setSoundVolume(sound.sound_volume);
        setSoundSettingsLoaded(true);
      }),
      load<AiSettings>("get_ai_settings", (ai) => {
        setAiSettings(ai);
        setAiSettingsLoaded(true);
      }),
      load<boolean>("get_autostart", setAutostart),
      load<boolean>("get_show_overlay", setShowOverlay),
      load<LanguageMode>("get_language", setLanguage),
      load<InputDeviceInfo[]>("get_input_devices", setInputDevices),
      load<string>("get_input_device", setInputDevice),
      load<string>("get_status", setStatus),
    ]).then((results) => {
      if (!mounted) return;
      if (results.some((result) => result.status === "rejected")) {
        setNotice("Some settings could not be loaded. Restart Wispr Local if this persists.");
      }
      setIsLoading(false);
    });

    const unlisten1 = listen<string>("status-changed", (event) => {
      setStatus(event.payload);
      if (event.payload !== "Recording") {
        setStreamingPreview("");
      }
    });

    const unlisten2 = listen<string[]>("history-changed", (event) => {
      setHistory(event.payload);
    });

    const unlisten3 = listen<string>("streaming-preview", (event) => {
      setStreamingPreview(event.payload);
    });

    let noticeTimer: ReturnType<typeof setTimeout> | undefined;
    const unlisten4 = listen<string>("transcription-empty", (event) => {
      const messages: Record<string, string> = {
        "too-short": "Recording too short — nothing captured",
        "no-speech": "No speech detected — try again",
        error: "Transcription failed — check logs",
      };
      setNotice(messages[event.payload] ?? "Nothing transcribed");
      clearTimeout(noticeTimer);
      noticeTimer = setTimeout(() => setNotice(""), 4000);
    });

    const unlisten5 = listen<string>("operation-notice", (event) => {
      setNotice(event.payload);
      clearTimeout(noticeTimer);
      noticeTimer = setTimeout(() => setNotice(""), 6000);
    });

    // The model loads on a background thread; this fires once it finishes (or
    // fails), flipping the footer indicator without a restart.
    const unlisten6 = listen<{ loaded: boolean; backend: string }>(
      "model-state-changed",
      (event) => {
        setModelLoaded(event.payload.loaded);
        setComputeBackend(event.payload.backend);
        setModelReported(true);
      },
    );

    return () => {
      mounted = false;
      void initialLoad;
      unlisten1.then((fn) => fn());
      unlisten2.then((fn) => fn());
      unlisten3.then((fn) => fn());
      unlisten4.then((fn) => fn());
      unlisten5.then((fn) => fn());
      unlisten6.then((fn) => fn());
      clearTimeout(noticeTimer);
      clearTimeout(copiedTimerRef.current);
    };
  }, []);

  // Text fields should feel immediate without rewriting settings.json on
  // every keystroke. Provider changes are included in the same short debounce.
  useEffect(() => {
    if (!aiSettingsLoaded) return;
    const timer = setTimeout(() => {
      invoke("set_ai_settings", { ai: aiSettings }).catch((error) =>
        setNotice(`Could not save AI settings: ${String(error)}`),
      );
    }, 450);
    return () => clearTimeout(timer);
  }, [aiSettings, aiSettingsLoaded]);

  useEffect(() => {
    if (!soundSettingsLoaded) return;
    const timer = setTimeout(() => {
      void saveSoundSettings(startSound, stopSound, soundVolume);
    }, 250);
    return () => clearTimeout(timer);
    // Sound paths are saved immediately by their own controls; this debounce
    // intentionally follows only the frequently-changing volume value.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [soundVolume, soundSettingsLoaded]);

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

  const updateAiSettings = (updates: Partial<AiSettings>) => {
    const newSettings = { ...aiSettings, ...updates };
    setAiSettings(newSettings);
  };

  const saveSoundSettings = (newStart: string, newStop: string, newVol: number) => {
    return invoke("set_sound_settings", {
      startSound: newStart,
      stopSound: newStop,
      soundVolume: newVol,
    }).catch((error) => {
      setNotice(`Could not save sound settings: ${String(error)}`);
    });
  };

  const pickSoundFile = async (which: "start" | "stop") => {
    const file = await open({
      multiple: false,
      filters: [{ name: "Audio", extensions: ["wav", "mp3", "ogg", "flac"] }],
    });
    if (typeof file === "string") {
      const path = file;
      if (which === "start") {
        setStartSound(path);
        void saveSoundSettings(path, stopSound, soundVolume);
      } else {
        setStopSound(path);
        void saveSoundSettings(startSound, path, soundVolume);
      }
    }
  };

  const clearSound = (which: "start" | "stop") => {
    if (which === "start") {
      setStartSound("");
      void saveSoundSettings("", stopSound, soundVolume);
    } else {
      setStopSound("");
      void saveSoundSettings(startSound, "", soundVolume);
    }
  };

  const handleVolumeChange = (vol: number) => {
    setSoundVolume(vol);
  };

  const testSound = (which: string) => {
    invoke("test_sound", { which }).catch((error) =>
      setNotice(`Could not play sound: ${String(error)}`),
    );
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
      .catch((error) => setNotice(`Copy failed: ${String(error)}`));
  };

  const clearHistory = async () => {
    if (!window.confirm("Clear all saved transcription history?")) return;
    try {
      await invoke("clear_history");
      setHistory([]);
      setNotice("History cleared");
    } catch (error) {
      setNotice(`Could not clear history: ${String(error)}`);
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
      setNotice(`Could not save setting: ${String(error)}`);
    }
  };

  const refreshInputDevices = async () => {
    setRefreshingDevices(true);
    try {
      setInputDevices(await invoke<InputDeviceInfo[]>("get_input_devices"));
    } catch (error) {
      setNotice(`Could not refresh microphones: ${String(error)}`);
    } finally {
      setRefreshingDevices(false);
    }
  };

  const hotkeyParts = hotkey.split("+");
  const isRecording = status === "Recording";
  const isTranscribing = status === "Transcribing";
  const isFormatting = status === "Formatting";
  const isInjecting = status === "Injecting";
  const isProcessing = isTranscribing || isFormatting || isInjecting;
  const hasError = status.startsWith("Error") || status.startsWith("Microphone error");

  return (
    <div className="app">
      <div className="header">
        <div className="logo">W</div>
        <span className="app-name">Wispr Local</span>
        <button
          type="button"
          className="settings-toggle"
          onClick={() => setShowSettings(!showSettings)}
          aria-label={showSettings ? "Return to dictation status" : "Open settings"}
          aria-expanded={showSettings}
        >
          {showSettings ? "Back" : "Settings"}
        </button>
      </div>

      {notice && (
        <div className="notice-banner" role="alert">
          {notice}
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
                ? status
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
                    setNotice(`Could not save language: ${String(error)}`);
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
                      setNotice(`Could not save microphone: ${String(error)}`);
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
                <button type="button" className="sound-btn" onClick={() => testSound("start")}>
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
                <button type="button" className="sound-btn" onClick={() => testSound("stop")}>
                  Test
                </button>
              </div>
            </div>

            <div className="volume-row">
              <span className="sound-label">Volume</span>
              <input
                type="range"
                min="0"
                max="100"
                value={Math.round(soundVolume * 100)}
                aria-label="Sound volume"
                aria-valuetext={`${Math.round(soundVolume * 100)} percent`}
                onChange={(e) => handleVolumeChange(Number(e.target.value) / 100)}
                className="volume-slider"
              />
              <span className="volume-value">{Math.round(soundVolume * 100)}%</span>
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
                    provider: e.target.value as AiSettings["provider"],
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
                  <input
                    id="openai-api-key"
                    className="setting-input"
                    type="password"
                    value={aiSettings.api_key}
                    onChange={(e) =>
                      updateAiSettings({ api_key: e.target.value })
                    }
                    placeholder="sk-..."
                    autoComplete="off"
                    spellCheck={false}
                  />
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
                  <input
                    id="claude-api-key"
                    className="setting-input"
                    type="password"
                    value={aiSettings.api_key}
                    onChange={(e) =>
                      updateAiSettings({ api_key: e.target.value })
                    }
                    placeholder="sk-ant-..."
                    autoComplete="off"
                    spellCheck={false}
                  />
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
                <div className="settings-note">
                  API key encrypted for your Windows account. Transcripts are sent only when AI formatting is enabled.
                </div>
              </>
            )}
          </div>
        </div>
      )}

      <div className="footer">
        <div
          className={`model-indicator ${
            isLoading || (!modelReported && !modelLoaded)
              ? ""
              : modelLoaded
                ? "ok"
                : "err"
          }`}
        >
          <span className="dot" />
          {isLoading || (!modelReported && !modelLoaded)
            ? "Checking model..."
            : modelLoaded
              ? `Model ready${computeBackend ? ` · ${computeBackend}` : ""}`
              : "Model not loaded"}
        </div>
        {!isLoading && modelReported && !modelLoaded && (
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
                    setNotice(`Could not open model folder: ${String(error)}`),
                  )
                }
              >
                Open folder
              </button>
              <span>Restart Wispr Local after adding the model.</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
