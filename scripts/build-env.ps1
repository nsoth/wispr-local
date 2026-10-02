# Dot-source this file to get the environment every cargo / tauri command needs
# on this machine: LLVM (bindgen), CMake, the CUDA toolkit and the MSVC
# developer shell. Usage from any PowerShell:
#
#   . .\scripts\build-env.ps1
#   cargo test --release --manifest-path src-tauri/Cargo.toml --lib
#
# The values mirror run-dev.ps1; keep both in sync when a toolkit moves.
# Deliberately does not touch $ErrorActionPreference: this file is dot-sourced,
# and a global "Stop" would turn cargo's stderr progress lines into terminating
# errors under Windows PowerShell 5.1.

$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
$llvmBin = "C:\Program Files\LLVM\bin"
$cmakeBin = "C:\Program Files\CMake\bin"
$vsRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$devShellDll = Join-Path $vsRoot "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"

foreach ($required in @($cudaRoot, $llvmBin, $cmakeBin, $devShellDll)) {
    if (-not (Test-Path $required)) {
        Write-Error "build-env: required path is missing: $required"
        return
    }
}

$env:LIBCLANG_PATH = $llvmBin
$env:CMAKE = Join-Path $cmakeBin "cmake.exe"
$env:CUDA_PATH = $cudaRoot
$env:CudaToolkitDir = "$cudaRoot\"
$env:PATH = "$env:PATH;$cmakeBin;$env:USERPROFILE\.cargo\bin;$cudaRoot\bin;$cudaRoot\bin\x64"
if (-not $env:RUST_LOG) { $env:RUST_LOG = "info" }

# MSVC compiler/linker for whisper.cpp and the CUDA kernels. Entering through
# the DevShell module (instead of Launch-VsDevShell.ps1) avoids the vswhere.exe
# dependency, and -SkipAutomaticLocation keeps the current directory.
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    Import-Module $devShellDll
    Enter-VsDevShell -VsInstallPath $vsRoot -DevCmdArguments "-arch=amd64 -host_arch=amd64" -SkipAutomaticLocation | Out-Null
}
