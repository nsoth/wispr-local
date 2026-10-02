//! Single-instance guard. Only the supervised child owns the named mutex; a
//! second launch starts its own supervisor/child pair, sees the guard and
//! exits cleanly instead of repeatedly crashing on the already-registered
//! global hotkey.
//!
//! A relaunch that arrives while the previous instance is still tearing
//! down (WebView2 shutdown can take seconds) waits for the mutex to be
//! released instead of giving up immediately, so "Quit, then double-click
//! the exe" works on the first try.

/// How long a new instance waits for a previous one to finish exiting.
const TEARDOWN_WAIT_MS: u32 = 10_000;

struct InstanceMutex(windows_sys::Win32::Foundation::HANDLE);

unsafe impl Send for InstanceMutex {}
unsafe impl Sync for InstanceMutex {}

impl Drop for InstanceMutex {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

/// Returns true when this process is the only running instance (and now
/// holds the guard), false when another instance already owns it and kept
/// it for the whole wait.
pub fn claim_single_instance() -> bool {
    use std::os::windows::ffi::OsStrExt;
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{
        GetLastError, ERROR_ALREADY_EXISTS, INVALID_HANDLE_VALUE, WAIT_ABANDONED, WAIT_OBJECT_0,
    };
    use windows_sys::Win32::System::Threading::{CreateMutexW, WaitForSingleObject};

    static INSTANCE_MUTEX: OnceLock<InstanceMutex> = OnceLock::new();
    let name: Vec<u16> = std::ffi::OsStr::new("Local\\WisprLocalAppInstance")
        .encode_wide()
        .chain(Some(0))
        .collect();
    // Initial owner: the creating process holds the mutex until it exits.
    let handle = unsafe { CreateMutexW(std::ptr::null(), 1, name.as_ptr()) };
    if handle.is_null() || handle == INVALID_HANDLE_VALUE {
        // Failure to create the guard should not make dictation unavailable.
        return true;
    }
    if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
        // Someone else owns it; wait for them to exit (or time out).
        let waited = unsafe { WaitForSingleObject(handle, TEARDOWN_WAIT_MS) };
        if waited != WAIT_OBJECT_0 && waited != WAIT_ABANDONED {
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(handle);
            }
            return false;
        }
        eprintln!("Previous instance released the guard; continuing");
    }
    INSTANCE_MUTEX.set(InstanceMutex(handle)).is_ok()
}
