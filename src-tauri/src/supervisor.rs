//! Crash supervisor.
//!
//! whisper.cpp's CUDA backend calls abort() on any CUDA error (GGML_ABORT →
//! fastfail 0xc0000409 in ucrtbase) — e.g. a VRAM allocation failing while
//! another app holds GPU memory. That kills the whole process and cannot be
//! caught in-process, so dictation used to silently die until the user
//! noticed and relaunched (crashes observed 2026-04-22 and 2026-07-22, both
//! inside ggml-cuda under `WhisperEngine::transcribe`).
//!
//! The fix: the process launched by the user (or autostart) is a thin
//! watchdog. It spawns the real app as a child with `WISPR_SUPERVISED=1`,
//! pipes the child's stdout/stderr into `<data_dir>/wispr.log` (this captures
//! the exact CUDA error text that used to vanish with the windows-subsystem
//! stderr), and restarts the child when it exits non-zero. A clean Quit
//! (exit code 0) ends the supervisor too.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

const CHILD_ENV: &str = "WISPR_SUPERVISED";
/// Escape hatch: set to run the app directly with no watchdog (debugging).
const NO_SUPERVISOR_ENV: &str = "WISPR_NO_SUPERVISOR";

const LOG_FILE: &str = "wispr.log";
const LOG_MAX_BYTES: u64 = 5 * 1024 * 1024;
/// A child that dies this quickly after launch is a startup crash, not a
/// mid-use GPU failure.
const FAST_CRASH: Duration = Duration::from_secs(30);
/// Give up after this many startup crashes in a row — restarting forever
/// would just burn CPU reloading a 1.5 GB model into a broken environment.
const MAX_FAST_CRASHES: u32 = 3;
const FORCE_CPU_ENV: &str = "WISPR_FORCE_CPU";
/// whisper.cpp's GGML_ABORT path on Windows terminates with fast-fail
/// STATUS_STACK_BUFFER_OVERRUN. Field crashes from ggml-cuda use this code.
const NATIVE_FAST_FAIL: u32 = 0xC000_0409;
/// Exit codes Windows hands a GUI process at logoff/shutdown:
/// DBG_TERMINATE_PROCESS when session teardown kills the child, and
/// STATUS_DLL_INIT_FAILED_LOGOFF for anything spawned while the window
/// station is already closing. Neither is a crash — restarting into a dying
/// session burns the fast-crash budget and pops a give-up MessageBox that
/// blocks shutdown (observed 2026-07-31).
const SESSION_END_EXIT_CODES: [u32; 2] = [0x4001_0004, 0xC000_026B];

/// True when this process is the supervised child that should run the app.
pub fn should_run_app() -> bool {
    std::env::var(CHILD_ENV).is_ok() || std::env::var(NO_SUPERVISOR_ENV).is_ok()
}

/// Supervise loop: spawn the app as a child, log its output, restart on
/// crash. Exits the process when the child quits cleanly or crashes too often.
pub fn run_supervisor() -> ! {
    let data_dir = crate::config::AppConfig::new().data_dir;
    let _ = std::fs::create_dir_all(&data_dir);
    let log_path = data_dir.join(LOG_FILE);

    let mut fast_crashes = 0u32;
    let mut force_cpu = false;
    let mut last_code;

    loop {
        rotate_log(&log_path);

        let exe = match std::env::current_exe() {
            Ok(p) => p,
            Err(e) => {
                sup_log(&log_path, &format!("cannot resolve own exe path: {e}"));
                std::process::exit(1);
            }
        };

        let started = Instant::now();
        let mut command = Command::new(&exe);
        command
            .args(std::env::args().skip(1))
            .env(CHILD_ENV, "1")
            // whisper-rs mirrors every whisper.cpp line (one per preview tick)
            // at info; keep only its warnings and errors — the CUDA error text
            // before an abort is logged at error level.
            .env(
                "RUST_LOG",
                std::env::var("RUST_LOG").unwrap_or_else(|_| "info,whisper_rs=warn".into()),
            )
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if force_cpu {
            command.env(FORCE_CPU_ENV, "1");
        }
        let mut child = match command.spawn() {
            Ok(c) => c,
            Err(e) => {
                sup_log(&log_path, &format!("failed to spawn child: {e}"));
                std::process::exit(1);
            }
        };
        sup_log(&log_path, &format!("app started (pid {})", child.id()));

        let drain_out = child
            .stdout
            .take()
            .map(|s| spawn_drain(s, log_path.clone()));
        let drain_err = child
            .stderr
            .take()
            .map(|s| spawn_drain(s, log_path.clone()));

        let status = child.wait();
        if let Some(t) = drain_out {
            let _ = t.join();
        }
        if let Some(t) = drain_err {
            let _ = t.join();
        }

        let code = status.ok().and_then(|s| s.code()).unwrap_or(-1);
        last_code = code;

        if code == 0 {
            sup_log(&log_path, "app quit cleanly");
            break;
        }
        if is_session_end(code) {
            sup_log(
                &log_path,
                &format!(
                    "session shutdown killed the app (exit code {:#x}) — exiting without restart",
                    code
                ),
            );
            last_code = 0;
            break;
        }
        sup_log(
            &log_path,
            &format!(
                "app crashed with exit code {:#x} after {}s — restarting",
                code,
                started.elapsed().as_secs()
            ),
        );

        if !force_cpu && is_native_fast_fail(code) {
            force_cpu = true;
            sup_log(
                &log_path,
                "native GPU-style fast-fail detected; next restart will use the CPU backend",
            );
        }

        if started.elapsed() < FAST_CRASH {
            fast_crashes += 1;
            if fast_crashes >= MAX_FAST_CRASHES {
                sup_log(&log_path, "giving up after repeated startup crashes");
                fatal_message_box(&format!(
                    "Wispr Local keeps crashing right after start.\n\nSee log:\n{}",
                    log_path.display()
                ));
                break;
            }
        } else {
            fast_crashes = 0;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    std::process::exit(last_code)
}

fn is_native_fast_fail(code: i32) -> bool {
    code as u32 == NATIVE_FAST_FAIL
}

fn is_session_end(code: i32) -> bool {
    SESSION_END_EXIT_CODES.contains(&(code as u32))
}

/// Copy child output into the log file line by line. In debug builds also
/// mirror to our own stderr so `tauri dev` still shows logs in the console.
fn spawn_drain(
    stream: impl std::io::Read + Send + 'static,
    log_path: PathBuf,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stream);
        let mut buf = Vec::with_capacity(512);
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&log_path)
                    {
                        let _ = f.write_all(&buf);
                    }
                    #[cfg(debug_assertions)]
                    {
                        let _ = std::io::stderr().write_all(&buf);
                    }
                }
            }
        }
    })
}

fn sup_log(log_path: &Path, msg: &str) {
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
    {
        let ts = humantime::format_rfc3339_seconds(SystemTime::now());
        let _ = writeln!(f, "[{ts} supervisor] {msg}");
    }
    #[cfg(debug_assertions)]
    eprintln!("[supervisor] {msg}");
}

fn rotate_log(log_path: &Path) {
    let too_big = std::fs::metadata(log_path)
        .map(|m| m.len() > LOG_MAX_BYTES)
        .unwrap_or(false);
    if too_big {
        let old = log_path.with_extension("log.old");
        let _ = std::fs::remove_file(&old);
        let _ = std::fs::rename(log_path, &old);
    }
}

#[cfg(windows)]
fn fatal_message_box(text: &str) {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        MessageBoxW, MB_ICONERROR, MB_OK, MB_SETFOREGROUND, MB_TOPMOST,
    };
    let wide = |s: &str| -> Vec<u16> { s.encode_utf16().chain(std::iter::once(0)).collect() };
    let text_w = wide(text);
    let title_w = wide("Wispr Local");
    unsafe {
        MessageBoxW(
            std::ptr::null_mut(),
            text_w.as_ptr(),
            title_w.as_ptr(),
            MB_OK | MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST,
        );
    }
}

#[cfg(not(windows))]
fn fatal_message_box(_text: &str) {}

#[cfg(test)]
mod tests {
    use super::{is_native_fast_fail, is_session_end, NATIVE_FAST_FAIL};

    #[test]
    fn recognizes_windows_native_fast_fail_exit_code() {
        assert!(is_native_fast_fail(NATIVE_FAST_FAIL as i32));
        assert!(!is_native_fast_fail(1));
    }

    #[test]
    fn session_end_codes_do_not_count_as_crashes() {
        assert!(is_session_end(0x4001_0004u32 as i32));
        assert!(is_session_end(0xC000_026Bu32 as i32));
        assert!(!is_session_end(NATIVE_FAST_FAIL as i32));
        assert!(!is_session_end(1));
        assert!(!is_session_end(0));
    }
}
