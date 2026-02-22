$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:CMAKE = "C:\Program Files\CMake\bin\cmake.exe"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
$env:CudaToolkitDir = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\"
$env:PATH = "$env:PATH;C:\Program Files\CMake\bin;$env:USERPROFILE\.cargo\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
$env:RUST_LOG = "info"

# Launch VS Developer Shell for MSVC compiler access (needed for CUDA build)
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64

Set-Location $PSScriptRoot
npx tauri dev
