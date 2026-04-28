# Wispr Local

Local, privacy-first voice-to-text dictation tool. Hold a hotkey, speak, and text appears wherever your cursor is. Powered by [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) with optional AI formatting via OpenAI or Claude.

## Features

- **Hold-to-dictate** global hotkey (customizable, default: `Ctrl+Shift+Space`)
- **Local Whisper.cpp** transcription — no internet required for the core dictation loop
- **CUDA GPU acceleration** for fast transcription
- **Real-time streaming preview** while recording
- **AI text formatting** (paragraphs, punctuation, bullet lists) via OpenAI or Claude (optional)
- **Automatic filler word removal** (English + Russian)
- **Recording overlay** — small always-on-top indicator above the taskbar (toggleable)
- **Animated tray icon** pulses while recording
- **Run on startup** — optional Windows autostart
- **Custom start/stop recording sounds** with volume control
- Built with **Tauri v2** (Rust + React)

## Language support

Whisper transcription is pinned to **Russian** (`language = "ru"`) by default. The medium model handles English code-switching inside a Russian utterance well — technical terms and mixed phrases come through correctly.

Why pinned and not auto-detect: Whisper's language detector mis-classifies Russian as Ukrainian / Belarusian / Polish 5–10% of the time on short utterances or with English code-switching. Auto-detect also makes Whisper vulnerable to a known initial-prompt echo hallucination on silent / low-signal segments. Pinning the language sidesteps both issues.

If you primarily speak a different language, change `set_language(Some("ru"))` in [`src-tauri/src/transcription/engine.rs`](src-tauri/src/transcription/engine.rs) to your language code (e.g. `"en"`, `"es"`).

## Requirements

- Windows 10/11
- [Rust](https://rustup.rs/) 1.77+
- [Node.js](https://nodejs.org/) 20+
- [CMake](https://cmake.org/) 3.5+
- [LLVM/Clang](https://releases.llvm.org/) (for whisper.cpp compilation)
- [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/downloads/) with "Desktop development with C++" workload
- NVIDIA GPU with CUDA toolkit (optional, for GPU acceleration)

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/nsoth/wispr-local.git
cd wispr-local
npm install
```

### 2. Set environment variables

whisper.cpp builds from source and needs CMake and LLVM:

```powershell
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:CMAKE = "C:\Program Files\CMake\bin\cmake.exe"
$env:Path = "C:\Program Files\CMake\bin;$env:Path"
```

Or use the included helper script:

```powershell
.\run-dev.ps1
```

### 3. Download a Whisper model

Download a GGML model to the app's data directory:

```powershell
# Create the models directory
$modelsDir = "$env:APPDATA\wispr-local\WisprLocal\data\models"
New-Item -ItemType Directory -Force -Path $modelsDir

# Download the medium model (~1.5 GB, best quality)
Invoke-WebRequest -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin" -OutFile "$modelsDir\ggml-medium.bin"

# Or download the base English model (~142 MB, faster, lower quality)
Invoke-WebRequest -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" -OutFile "$modelsDir\ggml-base.en.bin"
```

### 4. Run in development mode

```powershell
.\run-dev.ps1
# or manually:
npm run tauri dev
```

## Usage

1. The app starts minimized in the system tray
2. **Hold** `Ctrl+Shift+Space` (or your custom hotkey) and speak
3. **Release** the hotkey — your speech is transcribed and pasted into the focused text field
4. Right-click the tray icon for more options

### Settings

Click **Settings** in the app window to configure:

- **Hotkey** — rebind the hold-to-dictate global shortcut
- **Sounds** — custom start/stop recording sounds, volume control
- **Overlay** — show / hide the small recording indicator above the taskbar
- **Run on startup** — launch the app automatically when Windows starts
- **AI Formatting** — enable AI-powered text formatting:
  - **OpenAI** — uses GPT models, requires API key
  - **Claude** — uses Anthropic models, requires API key

  When AI formatting is off (default), transcribed text is pasted as-is after filler-word cleanup.

## Building for production

```powershell
npm run tauri build
```

This creates an installer in `src-tauri/target/release/bundle/`.

## Architecture

```
wispr-local/
├── src/                          # React frontend
│   ├── App.tsx                   # Main window UI (settings)
│   ├── Overlay.tsx               # Compact recording indicator
│   └── styles/                   # global.css, overlay.css
├── src-tauri/                    # Rust backend
│   └── src/
│       ├── lib.rs                # App setup, recording / transcription flow
│       ├── audio/                # Mic capture (cpal), resampling, buffer
│       ├── transcription/        # Whisper engine wrapper (pinned to "ru")
│       ├── formatting.rs         # AI formatting (OpenAI / Claude)
│       ├── system/               # Text injection, tray + animator, sounds
│       ├── autostart.rs          # Windows Run-key autostart
│       ├── settings.rs           # Persistent user settings
│       └── commands.rs           # Tauri IPC commands
```

## CUDA support

The project is configured with `whisper-rs = { features = ["cuda"] }` for GPU acceleration. If you don't have an NVIDIA GPU, change `Cargo.toml`:

```toml
whisper-rs = "0.15"  # remove features = ["cuda"]
```

## License

[MIT](LICENSE)
