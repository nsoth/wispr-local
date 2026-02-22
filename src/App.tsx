import { useState, useEffect, useCallback } from "react";
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

function App() {
  const [status, setStatus] = useState("Idle");
  const [lastTranscription, setLastTranscription] = useState("");
  const [streamingPreview, setStreamingPreview] = useState("");
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelsDir, setModelsDir] = useState("");
  const [hotkey, setHotkey] = useState("Ctrl+Shift+Space");
  const [isCapturingHotkey, setIsCapturingHotkey] = useState(false);
  const [hotkeyError, setHotkeyError] = useState("");
  const [startSound, setStartSound] = useState("");
  const [stopSound, setStopSound] = useState("");
  const [soundVolume, setSoundVolume] = useState(0.5);
  const [showSettings, setShowSettings] = useState(false);
  const [aiSettings, setAiSettings] = useState<AiSettings>({
    provider: "none",
    api_key: "",
    openai_model: "gpt-4o-mini",
    claude_model: "claude-sonnet-4-20250514",
    prompt: "",
  });

  useEffect(() => {
    invoke("is_model_loaded").then((loaded) => setModelLoaded(loaded as boolean));
    invoke("get_models_dir").then((dir) => setModelsDir(dir as string));
    invoke("get_hotkey").then((hk) => setHotkey(hk as string));
    invoke("get_last_transcription").then((t) => {
      const text = t as string;
      if (text) setLastTranscription(text);
    });
    invoke<SoundSettings>("get_sound_settings").then((s) => {
      setStartSound(s.start_sound);
      setStopSound(s.stop_sound);
      setSoundVolume(s.sound_volume);
    });
    invoke<AiSettings>("get_ai_settings").then((ai) => setAiSettings(ai));

    const unlisten1 = listen<string>("status-changed", (event) => {
      setStatus(event.payload);
      if (event.payload !== "Recording") {
        setStreamingPreview("");
      }
    });

    const unlisten2 = listen<string>("transcription-complete", (event) => {
      setLastTranscription(event.payload);
    });

    const unlisten3 = listen<string>("streaming-preview", (event) => {
      setStreamingPreview(event.payload);
    });

    return () => {
      unlisten1.then((fn) => fn());
      unlisten2.then((fn) => fn());
      unlisten3.then((fn) => fn());
    };
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
    invoke("set_ai_settings", { ai: newSettings });
  };

  const saveSoundSettings = (newStart: string, newStop: string, newVol: number) => {
    invoke("set_sound_settings", {
      startSound: newStart,
      stopSound: newStop,
      soundVolume: newVol,
    });
  };

  const pickSoundFile = async (which: "start" | "stop") => {
    const file = await open({
      multiple: false,
      filters: [{ name: "Audio", extensions: ["wav", "mp3", "ogg", "flac"] }],
    });
    if (file) {
      const path = typeof file === "string" ? file : file;
      if (which === "start") {
        setStartSound(path as string);
        saveSoundSettings(path as string, stopSound, soundVolume);
      } else {
        setStopSound(path as string);
        saveSoundSettings(startSound, path as string, soundVolume);
      }
    }
  };

  const clearSound = (which: "start" | "stop") => {
    if (which === "start") {
      setStartSound("");
      saveSoundSettings("", stopSound, soundVolume);
    } else {
      setStopSound("");
      saveSoundSettings(startSound, "", soundVolume);
    }
  };

  const handleVolumeChange = (vol: number) => {
    setSoundVolume(vol);
    saveSoundSettings(startSound, stopSound, vol);
  };

  const testSound = (which: string) => {
    invoke("test_sound", { which });
  };

  const fileName = (path: string) => {
    if (!path) return "";
    const parts = path.replace(/\\/g, "/").split("/");
    return parts[parts.length - 1];
  };

  const hotkeyParts = hotkey.split("+");
  const isRecording = status === "Recording";
  const isTranscribing = status === "Transcribing";
  const isFormatting = status === "Formatting";
  const isInjecting = status === "Injecting";
  const isProcessing = isTranscribing || isFormatting || isInjecting;

  return (
    <div className="app">
      <div className="header">
        <div className="logo">W</div>
        <span className="app-name">Wispr Local</span>
        <button
          className="settings-toggle"
          onClick={() => setShowSettings(!showSettings)}
          title="Settings"
        >
          {showSettings ? "Back" : "Settings"}
        </button>
      </div>

      {!showSettings ? (
        <>
          <div className="main-section">
            <div
              className={`mic-ring-container${
                isRecording ? " recording" : ""
              }${isProcessing ? " processing" : ""}`}
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
                >
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                  <line x1="12" y1="19" x2="12" y2="23" />
                  <line x1="8" y1="23" x2="16" y2="23" />
                </svg>
              </div>
            </div>

            <div className="status-label">
              {isRecording
                ? "Listening..."
                : isTranscribing
                ? "Transcribing..."
                : isFormatting
                ? "Formatting..."
                : isInjecting
                ? "Pasting..."
                : "Ready"}
            </div>

            {isRecording && streamingPreview && (
              <div className="streaming-preview">
                <div className="streaming-preview-text">{streamingPreview}</div>
              </div>
            )}

            <div className="hotkey-section">
              {isCapturingHotkey ? (
                <div className="hotkey-capture">
                  <span className="hotkey-capture-text">Press new hotkey...</span>
                  <button
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
                <div className="hotkey-error">{hotkeyError}</div>
              )}
            </div>
          </div>

          {lastTranscription && (
            <div className="transcript-card">
              <div className="transcript-label">Last transcription</div>
              <div className="transcript-text">{lastTranscription}</div>
            </div>
          )}
        </>
      ) : (
        <div className="settings-section">
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
                    <button className="sound-btn" onClick={() => clearSound("start")}>
                      Reset
                    </button>
                  </>
                ) : (
                  <span className="sound-file default">Built-in</span>
                )}
                <button className="sound-btn" onClick={() => pickSoundFile("start")}>
                  Browse
                </button>
                <button className="sound-btn" onClick={() => testSound("start")}>
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
                    <button className="sound-btn" onClick={() => clearSound("stop")}>
                      Reset
                    </button>
                  </>
                ) : (
                  <span className="sound-file default">Built-in</span>
                )}
                <button className="sound-btn" onClick={() => pickSoundFile("stop")}>
                  Browse
                </button>
                <button className="sound-btn" onClick={() => testSound("stop")}>
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
                onChange={(e) => handleVolumeChange(Number(e.target.value) / 100)}
                className="volume-slider"
              />
              <span className="volume-value">{Math.round(soundVolume * 100)}%</span>
            </div>
          </div>

          <div className="settings-group">
            <div className="settings-group-title">AI Formatting</div>

            <div className="setting-row">
              <span className="setting-label">Provider</span>
              <select
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
                  <span className="setting-label">API Key</span>
                  <input
                    className="setting-input"
                    type="password"
                    value={aiSettings.api_key}
                    onChange={(e) =>
                      updateAiSettings({ api_key: e.target.value })
                    }
                    placeholder="sk-..."
                  />
                </div>
                <div className="setting-row">
                  <span className="setting-label">Model</span>
                  <input
                    className="setting-input"
                    type="text"
                    value={aiSettings.openai_model}
                    onChange={(e) =>
                      updateAiSettings({ openai_model: e.target.value })
                    }
                    placeholder="gpt-4o-mini"
                  />
                </div>
              </>
            )}

            {aiSettings.provider === "claude" && (
              <>
                <div className="setting-row">
                  <span className="setting-label">API Key</span>
                  <input
                    className="setting-input"
                    type="password"
                    value={aiSettings.api_key}
                    onChange={(e) =>
                      updateAiSettings({ api_key: e.target.value })
                    }
                    placeholder="sk-ant-..."
                  />
                </div>
                <div className="setting-row">
                  <span className="setting-label">Model</span>
                  <input
                    className="setting-input"
                    type="text"
                    value={aiSettings.claude_model}
                    onChange={(e) =>
                      updateAiSettings({ claude_model: e.target.value })
                    }
                    placeholder="claude-sonnet-4-20250514"
                  />
                </div>
              </>
            )}

            {aiSettings.provider !== "none" && (
              <div className="setting-row prompt-row">
                <span className="setting-label">Prompt</span>
                <textarea
                  className="setting-textarea"
                  value={aiSettings.prompt}
                  onChange={(e) =>
                    updateAiSettings({ prompt: e.target.value })
                  }
                  rows={4}
                  placeholder="Custom formatting instructions..."
                />
              </div>
            )}
          </div>
        </div>
      )}

      <div className="footer">
        <div className={`model-indicator ${modelLoaded ? "ok" : "err"}`}>
          <span className="dot" />
          {modelLoaded ? "Model ready" : "Model not loaded"}
        </div>
        {!modelLoaded && (
          <div className="model-help">
            Download <code>ggml-base.en.bin</code> to:
            <span className="model-path">{modelsDir}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
