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
- **Microphone selection** with default-device fallback and disconnect recovery
- **Crash recovery** — a lightweight supervisor restarts the app after native CUDA failures
- **Automatic CPU recovery** after a native CUDA fast-fail
- **Encrypted API keys** using Windows Data Protection API (DPAPI)
- Built with **Tauri v2** (Rust + React)

## Language support

Whisper transcription defaults to **Auto (Russian / English)**. Auto mode runs a constrained two-language detection pass, biased toward Russian when Russian speech is present. This avoids accidental Ukrainian / Belarusian / Polish decoding while still handling fully English dictation. You can pin Russian or English in Settings.

Use a multilingual GGML model. English-only files ending in `.en.bin` cannot transcribe Russian and are intentionally skipped by model discovery.

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

# Recommended: large-v3-turbo (~1.6 GB, best speed/quality balance)
Invoke-WebRequest -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin" -OutFile "$modelsDir\ggml-large-v3-turbo.bin"

# Lower-memory fallback: multilingual base model (~142 MB)
Invoke-WebRequest -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin" -OutFile "$modelsDir\ggml-base.bin"
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
- **Microphone** — use the Windows default or choose a specific input device
- **Run on startup** — launch the app automatically when Windows starts
- **AI Formatting** — enable AI-powered text formatting:
  - **OpenAI** — uses GPT models, requires API key
  - **Claude** — uses Anthropic models, requires API key

  When AI formatting is off (default), transcribed text is pasted as-is after filler-word cleanup.

The core dictation loop stays local. Enabling OpenAI or Claude sends the transcript to that provider for formatting. API keys are encrypted for the current Windows user with DPAPI and migrated automatically from older plaintext settings. Other settings and the five-item transcription history are stored in the app data directory; history can be cleared from the main window.

## Building for production

```powershell
npm run tauri build
```

This creates an installer in `src-tauri/target/release/bundle/`.

Run `npm run check` for the frontend build and Rust unit tests.

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
│       ├── transcription/        # Whisper engine and constrained ru/en detection
│       ├── formatting.rs         # AI formatting (OpenAI / Claude)
│       ├── system/               # Text injection, tray + animator, sounds
│       ├── autostart.rs          # Windows Run-key autostart
│       ├── settings.rs           # Persistent user settings
│       ├── secrets.rs            # DPAPI-encrypted API-key storage
│       ├── supervisor.rs         # Native crash logging and restart watchdog
│       └── commands.rs           # Tauri IPC commands
```

## CUDA support

The project is configured with the `cuda` feature for GPU acceleration. If you don't have an NVIDIA GPU, keep native logging but remove CUDA in `Cargo.toml`:

```toml
whisper-rs = { version = "0.15", features = ["log_backend"] }
```

If CUDA initialization returns a regular error, the same process retries the model on CPU. If whisper.cpp terminates with its native CUDA fast-fail code, the supervisor restarts Wispr Local in CPU mode for the rest of that session. A clean application restart tries CUDA again.

## License

[MIT](LICENSE)
