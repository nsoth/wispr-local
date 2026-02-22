# Wispr Local

Local, privacy-first voice-to-text dictation tool. Hold a hotkey, speak, and text appears wherever your cursor is. Powered by [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) with optional AI formatting via Ollama, OpenAI, or Claude.

## Features

- **Hold-to-dictate** global hotkey (customizable, default: `Ctrl+Shift+Space`)
- **Local Whisper.cpp** transcription — no internet required
- **CUDA GPU acceleration** for fast transcription
- **Real-time streaming preview** while recording
- **AI text formatting** (paragraphs, punctuation, bullet lists) via Ollama / OpenAI / Claude
- **Automatic filler word removal** (English + Russian)
- **Custom start/stop recording sounds**
- **System tray app** — stays out of your way
- Built with **Tauri v2** (Rust + React)

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

- **Sounds** — custom start/stop recording sounds, volume control
- **AI Formatting** — enable AI-powered text formatting:
  - **Local (Ollama)** — runs on your machine, requires [Ollama](https://ollama.com/)
  - **OpenAI** — uses GPT models, requires API key
  - **Claude** — uses Anthropic models, requires API key

## Building for production

```powershell
npm run tauri build
```

This creates an installer in `src-tauri/target/release/bundle/`.

## Architecture

```
wispr-local/
├── src/                          # React frontend
│   ├── App.tsx                   # Main UI
│   └── styles/global.css         # Styles
├── src-tauri/                    # Rust backend
│   └── src/
│       ├── lib.rs                # App setup, recording/transcription flow
│       ├── audio/                # Mic capture (cpal), resampling, buffer
│       ├── transcription/        # Whisper engine wrapper
│       ├── formatting.rs         # AI formatting (Ollama/OpenAI/Claude)
│       ├── system/               # Text injection, tray, sounds
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
