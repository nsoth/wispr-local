//! Which window will receive the paste, and whether it should.
//!
//! The transcript is pasted seconds after the hotkey was released. If the
//! user has meanwhile clicked into another app (or into Wispr's own
//! settings), or the target is an elevated window that ignores synthesized
//! input, sending Ctrl+V would paste into the wrong place or silently do
//! nothing while the clipboard is restored over the text. In those cases the
//! transcript is left in the clipboard instead.

use std::time::{Duration, Instant};

/// Snapshot of the foreground window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PasteTarget {
    pub hwnd: isize,
    pub pid: u32,
    pub title: String,
    /// Executable file name ("Telegram.exe"), empty when unknown.
    pub exe: String,
}

impl PasteTarget {
    /// Short name for notices and the log.
    pub fn describe(&self) -> String {
        if !self.exe.is_empty() {
            self.exe.clone()
        } else if !self.title.is_empty() {
            self.title.clone()
        } else {
            "another window".to_string()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PasteDecision {
    Paste,
    /// Do not synthesize Ctrl+V; leave the text in the clipboard. The string
    /// is a short user-facing reason.
    CopyOnly(String),
}

/// Decide whether Ctrl+V may be sent. `then` is the window that was focused
/// when the recording stopped, `now` the one focused right before pasting.
pub fn decide_paste(
    then: Option<&PasteTarget>,
    now: Option<&PasteTarget>,
    own_pid: u32,
    target_elevated: bool,
) -> PasteDecision {
    let Some(now) = now else {
        return PasteDecision::CopyOnly("no window is focused (screen locked?)".to_string());
    };
    if now.pid == own_pid {
        return PasteDecision::CopyOnly("the Wispr Local window is focused".to_string());
    }
    if let Some(then) = then {
        if then.pid != now.pid {
            return PasteDecision::CopyOnly(format!("focus moved to {}", now.describe()));
        }
    }
    if target_elevated {
        return PasteDecision::CopyOnly(format!(
            "{} runs elevated and ignores synthesized input",
            now.describe()
        ));
    }
    PasteDecision::Paste
}

/// Poll `all_released` until it returns true or `timeout` passes. Returns
/// whether the keys were released in time.
pub fn wait_until_released(
    mut all_released: impl FnMut() -> bool,
    timeout: Duration,
    step: Duration,
) -> bool {
    let started = Instant::now();
    loop {
        if all_released() {
            return true;
        }
        if started.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(step);
    }
}

#[cfg(windows)]
pub fn foreground_target() -> Option<PasteTarget> {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowTextW, GetWindowThreadProcessId,
    };
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.is_null() {
            return None;
        }
        let mut pid: u32 = 0;
        GetWindowThreadProcessId(hwnd, &mut pid);
        let mut buf = [0u16; 256];
        let n = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32).max(0) as usize;
        Some(PasteTarget {
            hwnd: hwnd as isize,
            pid,
            title: String::from_utf16_lossy(&buf[..n]),
            exe: process_exe_name(pid),
        })
    }
}

#[cfg(not(windows))]
pub fn foreground_target() -> Option<PasteTarget> {
    None
}

#[cfg(windows)]
fn process_exe_name(pid: u32) -> String {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return String::new();
        }
        let mut buf = [0u16; 1024];
        let mut len = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(handle, PROCESS_NAME_WIN32, buf.as_mut_ptr(), &mut len);
        CloseHandle(handle);
        if ok == 0 {
            return String::new();
        }
        let path = String::from_utf16_lossy(&buf[..len as usize]);
        path.rsplit(['\\', '/'])
            .next()
            .unwrap_or_default()
            .to_string()
    }
}

/// Integrity level RID (0x1000 low, 0x2000 medium, 0x3000 high) of a process,
/// or None when it cannot be inspected.
#[cfg(windows)]
fn integrity_level(pid: u32) -> Option<u32> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::Security::{
        GetSidSubAuthority, GetSidSubAuthorityCount, GetTokenInformation, TokenIntegrityLevel,
        TOKEN_MANDATORY_LABEL, TOKEN_QUERY,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, OpenProcessToken, PROCESS_QUERY_LIMITED_INFORMATION,
    };
    unsafe {
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if process.is_null() {
            return None;
        }
        let mut token = std::ptr::null_mut();
        if OpenProcessToken(process, TOKEN_QUERY, &mut token) == 0 {
            CloseHandle(process);
            return None;
        }
        let mut len = 0u32;
        GetTokenInformation(
            token,
            TokenIntegrityLevel,
            std::ptr::null_mut(),
            0,
            &mut len,
        );
        let mut buf = vec![0u8; len.max(1) as usize];
        let ok = GetTokenInformation(
            token,
            TokenIntegrityLevel,
            buf.as_mut_ptr() as *mut _,
            len,
            &mut len,
        );
        let result = if ok == 0 || len == 0 {
            None
        } else {
            let label = &*(buf.as_ptr() as *const TOKEN_MANDATORY_LABEL);
            let sid = label.Label.Sid;
            let count = *GetSidSubAuthorityCount(sid);
            if count == 0 {
                None
            } else {
                Some(*GetSidSubAuthority(sid, u32::from(count) - 1))
            }
        };
        CloseHandle(token);
        CloseHandle(process);
        result
    }
}

/// Whether the process owning the target runs at a higher integrity level
/// than we do (then UIPI drops our synthesized input). Unknown levels are
/// treated as "not elevated" so a false positive can never block pasting.
#[cfg(windows)]
pub fn is_elevated_pid(pid: u32) -> bool {
    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    let own = unsafe { GetCurrentProcessId() };
    match (integrity_level(pid), integrity_level(own)) {
        (Some(target), Some(me)) => target > me,
        _ => false,
    }
}

#[cfg(not(windows))]
pub fn is_elevated_pid(_pid: u32) -> bool {
    false
}

/// Block until Shift / Ctrl / Alt / Win are physically up, so the paste is
/// Ctrl+V and not Ctrl+Shift+V (markdown preview in VS Code, paste-from-
/// history in JetBrains) when the user is still holding the hotkey's
/// modifiers at release time.
#[cfg(windows)]
pub fn wait_for_modifiers_released(timeout: Duration) -> bool {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, VK_CONTROL, VK_LWIN, VK_MENU, VK_RWIN, VK_SHIFT,
    };
    let modifiers = [VK_SHIFT, VK_CONTROL, VK_MENU, VK_LWIN, VK_RWIN];
    let all_up = || {
        modifiers
            .iter()
            .all(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) as u16 & 0x8000 == 0 })
    };
    wait_until_released(all_up, timeout, Duration::from_millis(10))
}

#[cfg(not(windows))]
pub fn wait_for_modifiers_released(_timeout: Duration) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::{decide_paste, wait_until_released, PasteDecision, PasteTarget};
    use std::time::{Duration, Instant};

    fn target(pid: u32, title: &str) -> PasteTarget {
        PasteTarget {
            hwnd: pid as isize * 10,
            pid,
            title: title.into(),
            exe: String::new(),
        }
    }

    #[test]
    fn pastes_into_the_window_focused_at_release() {
        let then = target(100, "Telegram");
        let now = target(100, "Telegram");
        assert_eq!(
            decide_paste(Some(&then), Some(&now), 7, false),
            PasteDecision::Paste
        );
    }

    #[test]
    fn same_app_different_window_is_fine() {
        // A dialog of the same process (same pid, other hwnd) keeps the paste.
        let then = target(100, "Telegram");
        let now = PasteTarget {
            hwnd: 999,
            ..target(100, "Telegram - dialog")
        };
        assert_eq!(
            decide_paste(Some(&then), Some(&now), 7, false),
            PasteDecision::Paste
        );
    }

    #[test]
    fn focus_moved_to_another_app_copies_only() {
        let then = target(100, "Telegram");
        let now = target(200, "VS Code");
        assert!(matches!(
            decide_paste(Some(&then), Some(&now), 7, false),
            PasteDecision::CopyOnly(_)
        ));
    }

    #[test]
    fn our_own_window_never_receives_the_paste() {
        let then = target(100, "Telegram");
        let now = target(7, "Wispr Local");
        assert!(matches!(
            decide_paste(Some(&then), Some(&now), 7, false),
            PasteDecision::CopyOnly(_)
        ));
    }

    #[test]
    fn no_foreground_window_or_elevated_target_copies_only() {
        let then = target(100, "Telegram");
        assert!(matches!(
            decide_paste(Some(&then), None, 7, false),
            PasteDecision::CopyOnly(_)
        ));
        let now = target(100, "Task Manager");
        assert!(matches!(
            decide_paste(Some(&then), Some(&now), 7, true),
            PasteDecision::CopyOnly(_)
        ));
    }

    #[test]
    fn unknown_origin_still_pastes_into_a_valid_target() {
        // Recording started from the tray with no capture of the origin.
        let now = target(100, "Telegram");
        assert_eq!(
            decide_paste(None, Some(&now), 7, false),
            PasteDecision::Paste
        );
    }

    #[test]
    fn wait_until_released_returns_as_soon_as_keys_are_up() {
        let mut polls = 0;
        let released = wait_until_released(
            || {
                polls += 1;
                polls >= 3
            },
            Duration::from_secs(1),
            Duration::from_millis(1),
        );
        assert!(released);
        assert_eq!(polls, 3);
    }

    #[test]
    fn wait_until_released_gives_up_after_the_timeout() {
        let started = Instant::now();
        let released = wait_until_released(
            || false,
            Duration::from_millis(40),
            Duration::from_millis(5),
        );
        assert!(!released);
        assert!(started.elapsed() >= Duration::from_millis(40));
        assert!(started.elapsed() < Duration::from_millis(400));
    }
}
