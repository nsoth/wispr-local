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
//!
//! Restart policy ([`RestartPolicy`]): a native fast-fail switches the NEXT
//! run to the CPU backend; if the CPU run then lives normally, the run after
//! it tries CUDA again (a transient VRAM shortage must not cost a whole day
//! of CPU transcription). Two fast-fails within ten minutes keep the CPU
//! for the session. Exit code [`RESTART_ON_GPU_CODE`] is a user request to
//! respawn on CUDA right away.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

const CHILD_ENV: &str = "WISPR_SUPERVISED";
/// Escape hatch: set to run the app directly with no watchdog (debugging).
const NO_SUPERVISOR_ENV: &str = "WISPR_NO_SUPERVISOR";

const LOG_FILE: &str = "wispr.log";
const LOG_MAX_BYTES: u64 = 5 * 1024 * 1024;
/// `wispr.log.1` .. `wispr.log.N` are kept; older ones are deleted.
const LOG_GENERATIONS: u32 = 3;
/// A child that dies this quickly after launch is a startup crash, not a
/// mid-use GPU failure.
const FAST_CRASH: Duration = Duration::from_secs(30);
/// Give up after this many startup crashes in a row — restarting forever
/// would just burn CPU reloading a 1.5 GB model into a broken environment.
const MAX_FAST_CRASHES: u32 = 3;
/// Two native fast-fails inside this window mean the GPU is really broken.
const CUDA_FAIL_WINDOW: Duration = Duration::from_secs(600);
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
/// The child asks to be respawned on CUDA (tray → "Restart on GPU").
pub const RESTART_ON_GPU_CODE: i32 = 0x5752;

/// True when this process is the supervised child that should run the app.
pub fn should_run_app() -> bool {
    std::env::var(CHILD_ENV).is_ok() || std::env::var(NO_SUPERVISOR_ENV).is_ok()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Leave the supervisor with this code.
    Exit(i32),
    /// Spawn the child again, on the CPU backend when `force_cpu`.
    Restart { force_cpu: bool, delay: Duration },
    /// Repeated startup crashes: stop and tell the user.
    GiveUp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuMode {
    Off,
    /// Only the next run uses the CPU.
    Once,
    /// Every run until the supervisor exits (or the user asks for GPU).
    Sticky,
}

/// Decides what to do after the child exits. Pure except for the clock
/// passed in, so the whole table is unit-tested.
#[derive(Debug)]
pub struct RestartPolicy {
    fast_crashes: u32,
    cpu_mode: CpuMode,
    cuda_fail_times: Vec<Instant>,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl RestartPolicy {
    pub fn new() -> Self {
        Self {
            fast_crashes: 0,
            cpu_mode: CpuMode::Off,
            cuda_fail_times: Vec::new(),
        }
    }

    /// Whether the next spawn must set `WISPR_FORCE_CPU`.
    pub fn force_cpu(&self) -> bool {
        self.cpu_mode != CpuMode::Off
    }

    pub fn on_exit(&mut self, code: i32, ran_for: Duration, now: Instant) -> Action {
        if code == 0 {
            return Action::Exit(0);
        }
        if is_session_end(code) {
            return Action::Exit(0);
        }
        if code == RESTART_ON_GPU_CODE {
            self.cpu_mode = CpuMode::Off;
            self.fast_crashes = 0;
            return Action::Restart {
                force_cpu: false,
                delay: Duration::ZERO,
            };
        }

        if is_native_fast_fail(code) {
            self.cuda_fail_times
                .retain(|t| now.duration_since(*t) < CUDA_FAIL_WINDOW);
            self.cuda_fail_times.push(now);
            self.cpu_mode = if self.cuda_fail_times.len() >= 2 {
                CpuMode::Sticky
            } else {
                CpuMode::Once
            };
        } else if self.cpu_mode == CpuMode::Once {
            // The CPU run ended for an unrelated reason: try CUDA again.
            self.cpu_mode = CpuMode::Off;
        }

        if ran_for < FAST_CRASH {
            self.fast_crashes += 1;
            if self.fast_crashes >= MAX_FAST_CRASHES {
                return Action::GiveUp;
            }
        } else {
            self.fast_crashes = 0;
        }
        Action::Restart {
            force_cpu: self.force_cpu(),
            delay: Duration::from_secs(1),
        }
    }

    /// Called when a run ended normally after living long enough: a one-shot
    /// CPU run that worked hands the next run back to CUDA.
    fn note_long_run(&mut self) {
        if self.cpu_mode == CpuMode::Once {
            self.cpu_mode = CpuMode::Off;
        }
    }
}

/// Build identity baked in by build.rs.
pub fn build_identity() -> String {
    let version = env!("CARGO_PKG_VERSION");
    let hash = option_env!("WISPR_GIT_HASH").unwrap_or("unknown");
    let dirty = option_env!("WISPR_GIT_DIRTY").unwrap_or("");
    let built = option_env!("WISPR_BUILD_TIME").unwrap_or("unknown");
    format!("v{version} ({hash}{dirty}, built {built})")
}

/// Supervise loop: spawn the app as a child, log its output, restart on
/// crash. Exits the process when the child quits cleanly or crashes too often.
pub fn run_supervisor() -> ! {
    let data_dir = crate::config::AppConfig::new().data_dir;
    let _ = std::fs::create_dir_all(&data_dir);
    let log_path = data_dir.join(LOG_FILE);

    let mut policy = RestartPolicy::new();
    let mut last_code;

    loop {
        rotate_if_large(&log_path);

        let exe = match std::env::current_exe() {
            Ok(p) => p,
            Err(e) => {
                sup_log(&log_path, &format!("cannot resolve own exe path: {e}"));
                std::process::exit(1);
            }
        };

        let args: Vec<String> = std::env::args().skip(1).collect();
        let started = Instant::now();
        let mut command = Command::new(&exe);
        command
            .args(&args)
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
        if policy.force_cpu() {
            command.env(FORCE_CPU_ENV, "1");
        }
        let mut child = match command.spawn() {
            Ok(c) => c,
            Err(e) => {
                sup_log(&log_path, &format!("failed to spawn child: {e}"));
                std::process::exit(1);
            }
        };
        sup_log(
            &log_path,
            &format!(
                "app started (pid {}, args {:?}, {}, backend {}, boot age {}s)",
                child.id(),
                args,
                build_identity(),
                if policy.force_cpu() { "CPU" } else { "auto" },
                boot_age_secs()
            ),
        );

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
        let ran_for = started.elapsed();
        if ran_for >= FAST_CRASH {
            policy.note_long_run();
        }

        match policy.on_exit(code, ran_for, Instant::now()) {
            Action::Exit(exit_code) => {
                if code == 0 {
                    sup_log(&log_path, "app quit cleanly");
                } else {
                    sup_log(
                        &log_path,
                        &format!(
                            "session shutdown killed the app (exit code {:#x}) — exiting without restart",
                            code
                        ),
                    );
                }
                last_code = exit_code;
                break;
            }
            Action::Restart { force_cpu, delay } => {
                if code == RESTART_ON_GPU_CODE {
                    sup_log(&log_path, "restart on GPU requested by the user");
                } else {
                    sup_log(
                        &log_path,
                        &format!(
                            "app crashed with exit code {:#x} after {}s — restarting{}",
                            code,
                            ran_for.as_secs(),
                            if force_cpu { " on the CPU backend" } else { "" }
                        ),
                    );
                    if is_native_fast_fail(code) {
                        sup_log(
                            &log_path,
                            "native GPU-style fast-fail detected; CUDA is retried after the next \
                             healthy run (or via tray → Restart on GPU)",
                        );
                    }
                }
                std::thread::sleep(delay);
            }
            Action::GiveUp => {
                sup_log(&log_path, "giving up after repeated startup crashes");
                fatal_message_box(&format!(
                    "Wispr Local keeps crashing right after start.\n\nSee log:\n{}",
                    log_path.display()
                ));
                break;
            }
        }
    }
    std::process::exit(last_code)
}

fn is_native_fast_fail(code: i32) -> bool {
    code as u32 == NATIVE_FAST_FAIL
}

fn is_session_end(code: i32) -> bool {
    SESSION_END_EXIT_CODES.contains(&(code as u32))
}

#[cfg(windows)]
fn boot_age_secs() -> u64 {
    use windows_sys::Win32::System::SystemInformation::GetTickCount64;
    unsafe { GetTickCount64() / 1000 }
}

#[cfg(not(windows))]
fn boot_age_secs() -> u64 {
    0
}

/// Serializes rotation against the two drain threads and the supervisor.
static ROTATE_LOCK: Mutex<()> = Mutex::new(());

/// Prefix lines that do not come from `log` (native whisper.cpp / rodio
/// output) with a timestamp so the log stays chronological.
pub fn prefix_raw_line(line: &[u8], timestamp: &str) -> Vec<u8> {
    if line.first() == Some(&b'[') || line.iter().all(|b| b.is_ascii_whitespace()) {
        return line.to_vec();
    }
    let mut out = Vec::with_capacity(line.len() + timestamp.len() + 12);
    out.extend_from_slice(b"[");
    out.extend_from_slice(timestamp.as_bytes());
    out.extend_from_slice(b" native] ");
    out.extend_from_slice(line);
    out
}

/// Copy child output into the log file line by line, rotating the file when
/// it outgrows the cap. In debug builds also mirror to our own stderr so
/// `tauri dev` still shows logs in the console.
fn spawn_drain(
    stream: impl std::io::Read + Send + 'static,
    log_path: PathBuf,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stream);
        let mut buf = Vec::with_capacity(512);
        let mut lines_since_check = 0u32;
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let ts = humantime::format_rfc3339_seconds(SystemTime::now()).to_string();
                    let line = prefix_raw_line(&buf, &ts);
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&log_path)
                    {
                        let _ = f.write_all(&line);
                    }
                    #[cfg(debug_assertions)]
                    {
                        let _ = std::io::stderr().write_all(&line);
                    }
                    lines_since_check += 1;
                    if lines_since_check >= 200 {
                        lines_since_check = 0;
                        rotate_if_large(&log_path);
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

fn rotate_if_large(log_path: &Path) {
    let too_big = std::fs::metadata(log_path)
        .map(|m| m.len() > LOG_MAX_BYTES)
        .unwrap_or(false);
    if too_big {
        let _guard = ROTATE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Re-check under the lock: another thread may have rotated already.
        let still_big = std::fs::metadata(log_path)
            .map(|m| m.len() > LOG_MAX_BYTES)
            .unwrap_or(false);
        if still_big {
            rotate_generations(log_path, LOG_GENERATIONS);
        }
    }
}

/// `wispr.log` → `wispr.log.1`, `.1` → `.2`, …; the oldest generation is
/// dropped. The crash and profiling history therefore survives several
/// rotations instead of one (the previous single `.old` was deleted on the
/// next rotation).
pub fn rotate_generations(log_path: &Path, generations: u32) {
    let generation = |n: u32| -> PathBuf {
        let name = log_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| LOG_FILE.to_string());
        log_path.with_file_name(format!("{name}.{n}"))
    };
    let _ = std::fs::remove_file(generation(generations));
    for n in (1..generations).rev() {
        let _ = std::fs::rename(generation(n), generation(n + 1));
    }
    let _ = std::fs::rename(log_path, generation(1));
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
    use super::{
        is_native_fast_fail, is_session_end, prefix_raw_line, rotate_generations, Action,
        RestartPolicy, NATIVE_FAST_FAIL, RESTART_ON_GPU_CODE,
    };
    use crate::config::test_dir;
    use std::time::{Duration, Instant};

    const FAST_FAIL: i32 = NATIVE_FAST_FAIL as i32;
    const LONG: Duration = Duration::from_secs(3600);
    const SHORT: Duration = Duration::from_secs(5);

    #[test]
    fn recognizes_windows_native_fast_fail_exit_code() {
        assert!(is_native_fast_fail(FAST_FAIL));
        assert!(!is_native_fast_fail(1));
    }

    #[test]
    fn session_end_codes_do_not_count_as_crashes() {
        assert!(is_session_end(0x4001_0004u32 as i32));
        assert!(is_session_end(0xC000_026Bu32 as i32));
        assert!(!is_session_end(FAST_FAIL));
        assert!(!is_session_end(1));
        assert!(!is_session_end(0));
        let mut p = RestartPolicy::new();
        assert_eq!(
            p.on_exit(0x4001_0004u32 as i32, SHORT, Instant::now()),
            Action::Exit(0)
        );
    }

    #[test]
    fn clean_quit_exits() {
        let mut p = RestartPolicy::new();
        assert_eq!(p.on_exit(0, LONG, Instant::now()), Action::Exit(0));
    }

    #[test]
    fn one_fast_fail_uses_cpu_once_then_returns_to_cuda() {
        let mut p = RestartPolicy::new();
        let now = Instant::now();
        assert_eq!(
            p.on_exit(FAST_FAIL, LONG, now),
            Action::Restart {
                force_cpu: true,
                delay: Duration::from_secs(1)
            }
        );
        assert!(p.force_cpu(), "next spawn is on the CPU");
        // The CPU run lived a long time and then quit for an unrelated reason
        // (e.g. a logoff-free crash): CUDA is tried again.
        p.note_long_run();
        assert!(!p.force_cpu());
        assert_eq!(
            p.on_exit(1, LONG, now + LONG),
            Action::Restart {
                force_cpu: false,
                delay: Duration::from_secs(1)
            }
        );
    }

    #[test]
    fn two_fast_fails_within_ten_minutes_keep_the_cpu() {
        let mut p = RestartPolicy::new();
        let now = Instant::now();
        p.on_exit(FAST_FAIL, LONG, now);
        p.note_long_run();
        assert!(!p.force_cpu());
        let action = p.on_exit(FAST_FAIL, LONG, now + Duration::from_secs(120));
        assert_eq!(
            action,
            Action::Restart {
                force_cpu: true,
                delay: Duration::from_secs(1)
            }
        );
        p.note_long_run();
        assert!(p.force_cpu(), "sticky after two fast-fails");
    }

    #[test]
    fn fast_fails_far_apart_stay_one_shot() {
        let mut p = RestartPolicy::new();
        let now = Instant::now();
        p.on_exit(FAST_FAIL, LONG, now);
        p.note_long_run();
        p.on_exit(FAST_FAIL, LONG, now + Duration::from_secs(3600));
        p.note_long_run();
        assert!(!p.force_cpu());
    }

    #[test]
    fn restart_on_gpu_request_clears_cpu_mode_and_is_not_a_crash() {
        let mut p = RestartPolicy::new();
        let now = Instant::now();
        p.on_exit(FAST_FAIL, LONG, now);
        p.on_exit(FAST_FAIL, LONG, now + SHORT);
        assert!(p.force_cpu());
        assert_eq!(
            p.on_exit(RESTART_ON_GPU_CODE, SHORT, now + Duration::from_secs(10)),
            Action::Restart {
                force_cpu: false,
                delay: Duration::ZERO
            }
        );
        assert!(!p.force_cpu());
        // Not counted as a fast crash: three quick restart requests never give up.
        p.on_exit(RESTART_ON_GPU_CODE, SHORT, now);
        p.on_exit(RESTART_ON_GPU_CODE, SHORT, now);
        assert_ne!(p.on_exit(RESTART_ON_GPU_CODE, SHORT, now), Action::GiveUp);
    }

    #[test]
    fn three_startup_crashes_give_up() {
        let mut p = RestartPolicy::new();
        let now = Instant::now();
        assert!(matches!(p.on_exit(1, SHORT, now), Action::Restart { .. }));
        assert!(matches!(p.on_exit(1, SHORT, now), Action::Restart { .. }));
        assert_eq!(p.on_exit(1, SHORT, now), Action::GiveUp);
    }

    #[test]
    fn a_long_run_resets_the_fast_crash_counter() {
        let mut p = RestartPolicy::new();
        let now = Instant::now();
        p.on_exit(1, SHORT, now);
        p.on_exit(1, SHORT, now);
        assert!(matches!(p.on_exit(1, LONG, now), Action::Restart { .. }));
        assert!(matches!(p.on_exit(1, SHORT, now), Action::Restart { .. }));
    }

    #[test]
    fn raw_lines_get_a_timestamp_and_log_lines_do_not() {
        assert_eq!(
            prefix_raw_line(
                b"whisper_init_state: kv self size\n",
                "2026-10-02T10:00:00Z"
            ),
            b"[2026-10-02T10:00:00Z native] whisper_init_state: kv self size\n".to_vec()
        );
        let already = b"[2026-10-02T10:00:00Z INFO  wispr] ok\n";
        assert_eq!(prefix_raw_line(already, "x"), already.to_vec());
        assert_eq!(prefix_raw_line(b"\n", "x"), b"\n".to_vec());
    }

    #[test]
    fn rotation_keeps_three_generations_in_order() {
        let dir = test_dir("rotate");
        let log = dir.join("wispr.log");
        for round in 1..=4u32 {
            std::fs::write(&log, format!("round {round}")).unwrap();
            rotate_generations(&log, 3);
        }
        assert!(!log.exists());
        assert_eq!(
            std::fs::read_to_string(dir.join("wispr.log.1")).unwrap(),
            "round 4"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("wispr.log.2")).unwrap(),
            "round 3"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("wispr.log.3")).unwrap(),
            "round 2"
        );
        assert!(
            !dir.join("wispr.log.4").exists(),
            "oldest generation dropped"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
