import { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { ask, open } from "@tauri-apps/plugin-dialog";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { EVENTS, IDLE_STATUS, type AppStatus, type LanguageDetected } from "./ipc";
import "./styles/global.css";

interface SoundSettings {
  start_sound: string;
  stop_sound: string;
  start_volume: number;
  stop_volume: number;
}

type AiProvider = "none" | "openai" | "claude";

// What the backend shows: never the key itself, only whether one is stored.
interface AiConnectionTest {
  latency_ms: number;
  model: string;
  sample: string;
}

interface StatsSummary {
  dictations: number;
  words: number;
  audio_s: number;
  no_result: number;
}

interface UsageStats {
  today: StatsSummary;
  week: StatsSummary;
}

const OPENAI_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o"];
const CLAUDE_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-5-5", "claude-opus-5-5"];

function fetchStats(): Promise<UsageStats> {
  const dayStart = new Date();
  dayStart.setHours(0, 0, 0, 0);
  return Promise.all([
    invoke<StatsSummary>("get_stats", { sinceMs: dayStart.getTime() }),
    invoke<StatsSummary>("get_stats", { sinceMs: Date.now() - 7 * 86_400_000 }),
  ]).then(([today, week]) => ({ today, week }));
}

function describeStats(s: StatsSummary): string {
  if (s.dictations === 0) return "no dictations";
  const parts = [
    `${s.dictations} ${s.dictations === 1 ? "dictation" : "dictations"}`,
    `${s.words} words`,
    `${(s.audio_s / 60).toFixed(s.audio_s < 600 ? 1 : 0)} min`,
  ];
  if (s.no_result > 0) {
    parts.push(`${Math.round((100 * s.no_result) / s.dictations)}% no result`);
  }
  return parts.join(" · ");
}

interface AiSettingsView {
  provider: AiProvider;
  enabled: boolean;
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
  settings_adjustments?: string[];
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

type PasteSuffix = "space" | "newline" | "none";

interface ReplacementRule {
  from: string;
  to: string;
  whole_word: boolean;
  case_insensitive: boolean;
}

interface TextSettings {
  voice_commands: boolean;
  paste_suffix: PasteSuffix;
  replacements: ReplacementRule[];
  restore_clipboard: boolean;
  history_limit: number;
}

type HotkeyMode = "hold" | "toggle" | "hybrid";
type HotkeyTarget = "main" | "cancel";

const HOTKEY_HINTS: Record<HotkeyMode, string> = {
  hold: "Hold to dictate, release to paste",
  toggle: "Press to start, press again to paste",
  hybrid: "Hold to dictate, or tap to go hands-free",
};

interface HistoryEntry {
  text: string;
  ts: number;
  target: string;
  lang: string;
  duration_s: number;
  pasted: boolean;
}

function relativeTime(ts: number, now: number): string {
  if (!ts) return "";
  const s = Math.max(0, Math.round((now - ts) / 1000));
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  const d = Math.round(h / 24);
  if (d < 7) return `${d} d ago`;
  return new Date(ts).toLocaleDateString();
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
  is_default: boolean;
}

// Mirrors the Rust LanguageMode enum (serde "auto"/"ru"/"en").
type LanguageMode = "auto" | "ru" | "en";

function App() {
  const [status, setStatus] = useState<AppStatus>(IDLE_STATUS);
  const [activeLanguage, setActiveLanguage] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [notice, setNoticeState] = useState<Notice | null>(null);
  const noticeTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const [diagnostics, setDiagnostics] = useState<string[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [historyFilter, setHistoryFilter] = useState("");
  const [stats, setStats] = useState<UsageStats | null>(null);
  const [aiTest, setAiTest] = useState<{
    status: "idle" | "running" | "ok" | "error";
    text: string;
  }>({ status: "idle", text: "" });
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const [clock, setClock] = useState(() => Date.now());
  const [streamingPreview, setStreamingPreview] = useState("");
  const [modelState, setModelState] = useState<ModelState>({ state: "loading" });
  const [textSettings, setTextSettings] = useState<TextSettings>({
    voice_commands: true,
    paste_suffix: "space",
    replacements: [],
    restore_clipboard: true,
    history_limit: 100,
  });
  const textRef = useRef<TextSettings>({
    voice_commands: true,
    paste_suffix: "space",
    replacements: [],
    restore_clipboard: true,
    history_limit: 100,
  });
  const textSaveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const [modelFiles, setModelFiles] = useState<ModelFileInfo[]>([]);
  const [modelsDir, setModelsDir] = useState("");
  const [appInfo, setAppInfo] = useState("");
  const [hotkey, setHotkey] = useState("Ctrl+Shift+Space");
  const [capturing, setCapturing] = useState<HotkeyTarget | null>(null);
  const [hotkeyError, setHotkeyError] = useState("");
  const [hotkeyMode, setHotkeyMode] = useState<HotkeyMode>("hold");
  const [cancelHotkey, setCancelHotkey] = useState("");
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
    enabled: true,
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
    fetchStats().then(setStats).catch(() => {});
    const RETRY_DELAYS = [300, 1000, 3000];
    const loadAll = (attempt: number) => Promise.allSettled([
      load<ModelState>("get_model_state", setModelState),
      load<TextSettings>("get_text_settings", (ts) => {
        setTextSettings(ts);
        textRef.current = ts;
      }),
      load<string>("get_models_dir", setModelsDir),
      load<string>("get_app_info", setAppInfo),
      load<string>("get_hotkey", setHotkey),
      load<HotkeyMode>("get_hotkey_mode", setHotkeyMode),
      load<string>("get_cancel_hotkey", setCancelHotkey),
      load<HistoryEntry[]>("get_history", setHistory),
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
        for (const adjustment of d.settings_adjustments ?? []) problems.push(adjustment);
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

    const unlisten2 = listen<HistoryEntry[]>(EVENTS.historyChanged, (event) => {
      setHistory(event.payload);
      setExpandedIndex(null);
      fetchStats().then(setStats).catch(() => {});
    });
    // Relative timestamps age without any other event arriving.
    const clockTimer = setInterval(() => setClock(Date.now()), 30_000);

    const unlisten3 = listen<string>(EVENTS.streamingPreview, (event) => {
      setStreamingPreview(event.payload);
    });

    const unlisten4 = listen<string>(EVENTS.transcriptionEmpty, (event) => {
      fetchStats().then(setStats).catch(() => {});
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

    const unlisten7 = listen<LanguageDetected>(EVENTS.languageDetected, (event) => {
      setActiveLanguage(event.payload.language.toUpperCase());
    });

    // Changes made from the tray reach the window through these events.
    const unlisten8 = listen(EVENTS.openSettings, () => {
      setShowSettings(true);
      void refreshModelFiles();
      void refreshInputDevices();
    });
    const unlisten9 = listen<LanguageMode>(EVENTS.languageModeChanged, (event) => {
      setLanguage(event.payload);
    });
    const unlisten10 = listen(EVENTS.aiSettingsChanged, () => {
      void invoke<AiSettingsView>("get_ai_settings")
        .then((ai) => {
          aiRef.current.view = ai;
          setAiSettings(ai);
        })
        .catch(() => undefined);
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
      unlisten7.then((fn) => fn());
      unlisten8.then((fn) => fn());
      unlisten9.then((fn) => fn());
      unlisten10.then((fn) => fn());
      clearInterval(clockTimer);
      clearTimeout(noticeTimerRef.current);
      clearTimeout(copiedTimerRef.current);
      clearTimeout(soundSaveTimer.current);
      clearTimeout(aiSaveTimer.current);
      clearTimeout(textSaveTimer.current);
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
    if (code === "Insert") return "Insert";
    if (code === "Home") return "Home";
    if (code === "End") return "End";
    if (code === "PageUp") return "PageUp";
    if (code === "PageDown") return "PageDown";
    if (code === "Pause") return "Pause";
    if (code === "ScrollLock") return "ScrollLock";
    if (code === "CapsLock") return "CapsLock";
    if (code === "NumLock") return "NumLock";
    if (code.startsWith("Numpad")) return code;
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

  // Keys that never type, so they may be a hotkey on their own.
  const isSafeBareKey = (key: string) =>
    /^F(1[3-9]|2[0-4])$/.test(key) || ["Pause", "ScrollLock", "CapsLock"].includes(key);

  const stopCapture = () => {
    setCapturing(null);
    void invoke("end_hotkey_capture").catch(() => undefined);
  };

  const startCapture = (target: HotkeyTarget) => {
    setHotkeyError("");
    setCapturing(target);
    // The live shortcut is ignored while capturing, so pressing the current
    // combination to confirm it does not start a recording.
    void invoke("begin_hotkey_capture").catch(() => undefined);
  };

  const handleHotkeyCapture = useCallback(
    (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (!capturing) return;

      const keyName = keyCodeToName(e);
      if (!keyName) return;

      if (keyName === "Escape" && !e.ctrlKey && !e.shiftKey && !e.altKey && !e.metaKey) {
        stopCapture();
        return;
      }

      const hasModifier = e.ctrlKey || e.shiftKey || e.altKey || e.metaKey;
      if (!hasModifier && !isSafeBareKey(keyName)) {
        setHotkeyError(
          "Include Ctrl, Shift, Alt or Win, or use a key that never types (F13–F24, Pause, ScrollLock)",
        );
        return;
      }

      const parts: string[] = [];
      if (e.ctrlKey) parts.push("Ctrl");
      if (e.shiftKey) parts.push("Shift");
      if (e.altKey) parts.push("Alt");
      if (e.metaKey) parts.push("Win");
      parts.push(keyName);
      const newHotkey = parts.join("+");
      const target = capturing;
      stopCapture();
      setHotkeyError("");

      if (target === "main") {
        invoke("set_hotkey", { hotkey: newHotkey })
          .then(() => setHotkey(newHotkey))
          .catch((err) => setHotkeyError(String(err)));
      } else {
        invoke<string>("set_cancel_hotkey", { hotkey: newHotkey })
          .then((saved) => setCancelHotkey(saved))
          .catch((err) => setHotkeyError(String(err)));
      }
    },
    [capturing],
  );

  useEffect(() => {
    if (capturing) {
      window.addEventListener("keydown", handleHotkeyCapture, true);
      return () => window.removeEventListener("keydown", handleHotkeyCapture, true);
    }
  }, [capturing, handleHotkeyCapture]);

  // Escape leaves Settings or hides the window; Ctrl+, opens Settings.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (capturing) return;
      const target = e.target as HTMLElement | null;
      const typing =
        target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA");
      if (e.key === "Escape") {
        if (typing && target) {
          target.blur();
          return;
        }
        if (showSettings) setShowSettings(false);
        else void getCurrentWindow().hide();
      } else if (e.key === "," && e.ctrlKey) {
        e.preventDefault();
        setShowSettings(true);
        void refreshModelFiles();
        void refreshInputDevices();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // refreshModelFiles / refreshInputDevices are stable arrow functions.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [capturing, showSettings]);

  const toggleHandsFree = () => {
    const command = isRecording ? "stop_recording" : "start_hands_free";
    invoke(command).catch((error) => setError(String(error)));
  };

  const resetPrompt = () => {
    invoke<string>("get_default_prompt")
      .then((prompt) => updateAiSettings({ prompt }))
      .catch((error) => setError(String(error)));
  };

  // Saves a key that is still only typed, then sends one tiny request.
  const testAiConnection = () => {
    setAiTest({ status: "running", text: "" });
    const draft = aiRef.current.draft;
    const unsaved = draft.openai.trim() !== "" || draft.claude.trim() !== "";
    const ready = unsaved ? flushAiSave() : Promise.resolve();
    ready
      .then(() => invoke<AiConnectionTest>("test_ai_connection"))
      .then((r) =>
        setAiTest({
          status: "ok",
          text: `OK · ${r.latency_ms} ms · ${r.model} · “${r.sample.slice(0, 60)}”`,
        }),
      )
      .catch((error) => setAiTest({ status: "error", text: String(error) }));
  };

  const openPath = (kind: "models" | "data" | "log" | "recordings") => {
    invoke("open_path", { kind }).catch((error) =>
      setError(`Could not open the folder: ${String(error)}`),
    );
  };

  const changeHotkeyMode = (mode: HotkeyMode) => {
    const previous = hotkeyMode;
    setHotkeyMode(mode);
    invoke("set_hotkey_mode", { mode }).catch((error) => {
      setHotkeyMode(previous);
      setError(`Could not save the hotkey mode: ${String(error)}`);
    });
  };

  const clearCancelHotkey = () => {
    invoke<string>("set_cancel_hotkey", { hotkey: "" })
      .then(() => setCancelHotkey(""))
      .catch((err) => setHotkeyError(String(err)));
  };

  // AI settings are saved 450 ms after the last edit, and only after an edit:
  // the initial load never writes, so a stored key can never be wiped by a
  // page that has not seen it.
  const flushAiSave = () => {
    clearTimeout(aiSaveTimer.current);
    const { view, draft } = aiRef.current;
    const update = {
      provider: view.provider,
      enabled: view.enabled,
      openai_model: view.openai_model,
      claude_model: view.claude_model,
      prompt: view.prompt,
      openai_api_key: draft.openai.trim() ? draft.openai.trim() : undefined,
      claude_api_key: draft.claude.trim() ? draft.claude.trim() : undefined,
    };
    return invoke<AiSettingsView>("set_ai_settings", { update })
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
      enabled: view.enabled,
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
    const yes = await ask("Clear all saved transcription history?", {
      title: "Wispr Local",
      kind: "warning",
      okLabel: "Clear",
      cancelLabel: "Keep",
    });
    if (!yes) return;
    try {
      await invoke("clear_history");
      setHistory([]);
      setNotice("History cleared");
    } catch (error) {
      setError(`Could not clear history: ${String(error)}`);
    }
  };

  const pasteHistoryItem = (index: number) => {
    invoke<string>("paste_history_item", { index })
      .then((result) =>
        setNotice(result === "pasted" ? "Pasted into the original window" : "Copied to clipboard (window is gone)"),
      )
      .catch((error) => setError(`Paste failed: ${String(error)}`));
  };

  const visibleHistory = history
    .map((entry, index) => ({ entry, index }))
    .filter(({ entry }) =>
      historyFilter.trim() ? entry.text.toLowerCase().includes(historyFilter.trim().toLowerCase()) : true,
    );

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

  // Text post-processing settings: saved 400 ms after the last edit.
  const updateTextSettings = (updates: Partial<TextSettings>) => {
    const next = { ...textRef.current, ...updates };
    textRef.current = next;
    setTextSettings(next);
    clearTimeout(textSaveTimer.current);
    textSaveTimer.current = setTimeout(() => {
      invoke("set_text_settings", { update: textRef.current }).catch((error) =>
        setError(`Could not save text settings: ${String(error)}`),
      );
    }, 400);
  };

  const updateRule = (index: number, changes: Partial<ReplacementRule>) => {
    const replacements = textRef.current.replacements.map((r, i) =>
      i === index ? { ...r, ...changes } : r,
    );
    updateTextSettings({ replacements });
  };

  const addRule = () => {
    updateTextSettings({
      replacements: [
        ...textRef.current.replacements,
        { from: "", to: "", whole_word: true, case_insensitive: true },
      ],
    });
  };

  const removeRule = (index: number) => {
    updateTextSettings({
      replacements: textRef.current.replacements.filter((_, i) => i !== index),
    });
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
  const isMicError = hasError && status.code === "mic";
  const errorText = isMicError
    ? "Microphone unavailable"
    : status.message || "Something went wrong";

  const retryMicrophone = () => {
    invoke<string>("probe_input_device")
      .then((device) => setNotice(`Microphone OK: ${device}`))
      .catch((error) => setError(`Microphone still unavailable: ${String(error)}`));
  };

  const openMicrophoneSettings = () => {
    setShowSettings(true);
    void refreshInputDevices();
    setTimeout(() => document.getElementById("microphone-select")?.focus(), 50);
  };

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
            if (next) {
              void refreshModelFiles();
              void refreshInputDevices();
            }
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
            <button
              type="button"
              className={`mic-ring-container${
                isRecording ? " recording" : ""
              }${isProcessing ? " processing" : ""}${hasError ? " error" : ""}`}
              onClick={toggleHandsFree}
              disabled={isProcessing || (!modelReady && !modelLoading)}
              aria-label={isRecording ? "Stop and paste" : "Start hands-free dictation"}
              title={
                isRecording
                  ? "Stop and paste"
                  : "Start a hands-free recording (focus the target app first, then stop from here, the tray or the hotkey)"
              }
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
            </button>

            <div className={`status-label${hasError ? " error" : ""}`} role="status" aria-live="polite">
              {isLoading
                ? "Loading..."
                : isRecording
                ? `Listening...${activeLanguage ? ` ${activeLanguage}` : ""}`
                : isTranscribing
                ? `Transcribing...${activeLanguage ? ` ${activeLanguage}` : ""}`
                : isFormatting
                ? "Formatting..."
                : isInjecting
                ? "Pasting..."
                : hasError
                ? errorText
                : "Ready"}
            </div>

            {isMicError && (
              <div className="mic-error">
                <div className="mic-error-detail">{status.message}</div>
                <div className="mic-error-actions">
                  <button type="button" className="sound-btn" onClick={openMicrophoneSettings}>
                    Choose microphone
                  </button>
                  <button type="button" className="sound-btn" onClick={retryMicrophone}>
                    Retry
                  </button>
                </div>
              </div>
            )}

            {isRecording && streamingPreview && (
              <div className="streaming-preview" aria-live="polite">
                <div className="streaming-preview-text">{streamingPreview}</div>
              </div>
            )}

            <div className="hotkey-section">
              {capturing === "main" ? (
                <div className="hotkey-capture">
                  <span className="hotkey-capture-text">Press new hotkey...</span>
                  <button type="button" className="hotkey-cancel-btn" onClick={stopCapture}>
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
                      onClick={() => startCapture("main")}
                      title="Change hotkey"
                    >
                      Change
                    </button>
                  </div>
                  <div className="hotkey-desc">{HOTKEY_HINTS[hotkeyMode]}</div>
                </>
              )}
              {hotkeyError && (
                <div className="hotkey-error" role="alert">{hotkeyError}</div>
              )}
            </div>
          </div>

          {stats && (stats.today.dictations > 0 || stats.week.dictations > 0) && (
            <div className="stats-card" aria-label="Usage">
              <div className="stats-row">
                <span className="stats-label">Today</span>
                <span className="stats-value">{describeStats(stats.today)}</span>
              </div>
              <div className="stats-row">
                <span className="stats-label">7 days</span>
                <span className="stats-value">{describeStats(stats.week)}</span>
              </div>
            </div>
          )}

          {history.length > 0 && (
            <div className="transcript-card">
              <div className="transcript-heading">
                <div className="transcript-label">History · {history.length}</div>
                <input
                  type="search"
                  className="history-search"
                  placeholder="Search…"
                  aria-label="Search history"
                  value={historyFilter}
                  onChange={(e) => setHistoryFilter(e.target.value)}
                />
                <button type="button" className="history-clear-btn" onClick={clearHistory}>
                  Clear
                </button>
              </div>
              <div className="history-list">
                {visibleHistory.length === 0 && (
                  <div className="history-empty">Nothing matches.</div>
                )}
                {visibleHistory.map(({ entry, index }) => (
                  <div
                    className={`history-item${expandedIndex === index ? " expanded" : ""}`}
                    key={`${entry.ts}-${index}`}
                  >
                    <div className="history-body">
                      <div className="history-meta">
                        <span>{relativeTime(entry.ts, clock) || "earlier"}</span>
                        {entry.target && <span>· {entry.target.replace(/\.exe$/i, "")}</span>}
                        {entry.lang && <span className="history-lang">{entry.lang.toUpperCase()}</span>}
                        {!entry.pasted && <span className="history-flag">not pasted</span>}
                      </div>
                      <button
                        type="button"
                        className="history-text"
                        lang={/[\u0400-\u04FF]/.test(entry.text) ? "ru" : "en"}
                        onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
                        title={expandedIndex === index ? "Collapse" : "Expand"}
                      >
                        {entry.text}
                      </button>
                    </div>
                    <div className="history-actions">
                      <button
                        type="button"
                        className={`history-copy-btn${copiedIndex === index ? " copied" : ""}`}
                        onClick={() => copyHistoryItem(entry.text, index)}
                        title="Copy to clipboard"
                      >
                        {copiedIndex === index ? "Copied" : "Copy"}
                      </button>
                      <button
                        type="button"
                        className="history-copy-btn"
                        onClick={() => pasteHistoryItem(index)}
                        title="Paste again into the original window"
                      >
                        Paste
                      </button>
                    </div>
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
              <label className="setting-label" htmlFor="hotkey-mode">Hotkey mode</label>
              <select
                id="hotkey-mode"
                className="setting-select"
                value={hotkeyMode}
                onChange={(e) => changeHotkeyMode(e.target.value as HotkeyMode)}
              >
                <option value="hold">Hold to talk</option>
                <option value="toggle">Toggle (press to start / stop)</option>
                <option value="hybrid">Hold, or tap for hands-free</option>
              </select>
            </div>
            <div className="setting-row">
              <span className="setting-label">Cancel key</span>
              <div className="device-controls">
                {capturing === "cancel" ? (
                  <span className="hotkey-capture-text">Press a combination…</span>
                ) : (
                  <span className="sound-file" title={cancelHotkey || "disabled"}>
                    {cancelHotkey || "Disabled"}
                  </span>
                )}
                <button
                  type="button"
                  className="sound-btn"
                  onClick={() => (capturing === "cancel" ? stopCapture() : startCapture("cancel"))}
                >
                  {capturing === "cancel" ? "Cancel" : "Change"}
                </button>
                {cancelHotkey && capturing !== "cancel" && (
                  <button type="button" className="sound-btn" onClick={clearCancelHotkey}>
                    Disable
                  </button>
                )}
              </div>
            </div>
            <div className="settings-note">
              Cancel discards the current recording, or skips the paste of the one being
              transcribed. The overlay's × does the same.
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
                      {device.name}
                      {device.is_default ? " (default)" : ""} ·{" "}
                      {Math.round(device.sample_rate / 1000)} kHz ·{" "}
                      {device.channels === 1 ? "mono" : "stereo"}
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
            <div className="settings-group-title">Text</div>
            <div className="setting-row">
              <span className="setting-label" id="voice-commands-label">
                Spoken line breaks
              </span>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  aria-labelledby="voice-commands-label"
                  checked={textSettings.voice_commands}
                  onChange={(e) => updateTextSettings({ voice_commands: e.target.checked })}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
            <div className="settings-note">
              Say "новая строка" / "new line" or "новый абзац" / "new paragraph" between pauses.
            </div>
            <div className="setting-row">
              <label className="setting-label" htmlFor="paste-suffix">After paste</label>
              <select
                id="paste-suffix"
                className="setting-select"
                value={textSettings.paste_suffix}
                onChange={(e) => updateTextSettings({ paste_suffix: e.target.value as PasteSuffix })}
              >
                <option value="space">Add a space</option>
                <option value="newline">Add a line break</option>
                <option value="none">Nothing</option>
              </select>
            </div>
            <div className="setting-row">
              <span className="setting-label" id="restore-clipboard-label">
                Restore clipboard after paste
              </span>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  aria-labelledby="restore-clipboard-label"
                  checked={textSettings.restore_clipboard}
                  onChange={(e) => updateTextSettings({ restore_clipboard: e.target.checked })}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
            <div className="settings-note">
              Off keeps the transcript in the clipboard for a manual Ctrl+V.
            </div>
            <div className="setting-row">
              <label className="setting-label" htmlFor="history-limit">Keep history</label>
              <select
                id="history-limit"
                className="setting-select"
                value={String(textSettings.history_limit)}
                onChange={(e) => updateTextSettings({ history_limit: Number(e.target.value) })}
              >
                <option value="0">Nothing (private)</option>
                <option value="20">Last 20</option>
                <option value="100">Last 100</option>
                <option value="500">Last 500</option>
              </select>
            </div>
            <div className="dict-heading">
              <span className="setting-label">Dictionary</span>
              <button type="button" className="sound-btn" onClick={addRule}>
                Add rule
              </button>
            </div>
            <div className="settings-note">
              Replace how Whisper spells a term with how you write it. W = whole words only, Aa =
              match case.
            </div>
            {textSettings.replacements.map((rule, index) => (
              <div className="dict-row" key={index}>
                <input
                  className="setting-input"
                  type="text"
                  value={rule.from}
                  placeholder="heard as…"
                  aria-label="Text to replace"
                  spellCheck={false}
                  onChange={(e) => updateRule(index, { from: e.target.value })}
                />
                <span className="dict-arrow">→</span>
                <input
                  className="setting-input"
                  type="text"
                  value={rule.to}
                  placeholder="write as…"
                  aria-label="Replacement"
                  spellCheck={false}
                  onChange={(e) => updateRule(index, { to: e.target.value })}
                />
                <label className="dict-flag" title="Whole words only">
                  <input
                    type="checkbox"
                    checked={rule.whole_word}
                    onChange={(e) => updateRule(index, { whole_word: e.target.checked })}
                  />
                  W
                </label>
                <label className="dict-flag" title="Match case">
                  <input
                    type="checkbox"
                    checked={!rule.case_insensitive}
                    onChange={(e) => updateRule(index, { case_insensitive: !e.target.checked })}
                  />
                  Aa
                </label>
                <button
                  type="button"
                  className="dict-remove"
                  onClick={() => removeRule(index)}
                  aria-label="Remove rule"
                  title="Remove rule"
                >
                  ×
                </button>
              </div>
            ))}
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
                  openPath("models")
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
                    list="openai-models"
                  />
                  <datalist id="openai-models">
                    {OPENAI_MODELS.map((m) => (
                      <option value={m} key={m} />
                    ))}
                  </datalist>
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
                    list="claude-models"
                  />
                  <datalist id="claude-models">
                    {CLAUDE_MODELS.map((m) => (
                      <option value={m} key={m} />
                    ))}
                  </datalist>
                </div>
              </>
            )}

            {aiSettings.provider !== "none" && (
              <>
                <div className="setting-row">
                  <span className="setting-label" id="ai-enabled-label">Formatting on</span>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      aria-labelledby="ai-enabled-label"
                      checked={aiSettings.enabled}
                      onChange={(e) => updateAiSettings({ enabled: e.target.checked })}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
                <div className="setting-row">
                  <span className="setting-label">Connection</span>
                  <div className="key-controls">
                    {aiTest.text && (
                      <span
                        className={`ai-test-result${aiTest.status === "error" ? " error" : ""}`}
                        title={aiTest.text}
                      >
                        {aiTest.text}
                      </span>
                    )}
                    <button
                      type="button"
                      className="sound-btn"
                      onClick={testAiConnection}
                      disabled={aiTest.status === "running"}
                    >
                      {aiTest.status === "running" ? "Testing…" : "Test"}
                    </button>
                  </div>
                </div>
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
                  <button type="button" className="sound-btn" onClick={resetPrompt}>
                    Reset prompt to default
                  </button>
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

      {showSettings && (
        <div className="settings-group about-group">
          <div className="settings-group-title">About</div>
          <div className="settings-note">Wispr Local {appInfo || "(build info unavailable)"}</div>
          <div className="about-actions">
            <button type="button" className="sound-btn" onClick={() => openPath("data")}>
              Open data folder
            </button>
            <button type="button" className="sound-btn" onClick={() => openPath("log")}>
              Open log
            </button>
            <button type="button" className="sound-btn" onClick={() => openPath("recordings")}>
              Open recordings
            </button>
          </div>
          <div className="settings-note">
            The last 30 recordings are kept as WAV files; the tray menu can re-transcribe the
            newest one if a result looks wrong.
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
                  openPath("models")
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
