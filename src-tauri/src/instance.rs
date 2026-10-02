//! Single-instance guard. Only the supervised child owns the named mutex; a
//! second launch starts its own supervisor/child pair, sees the guard and
//! exits cleanly instead of repeatedly crashing on the already-registered
//! global hotkey.

#[cfg(windows)]
struct InstanceMutex(windows_sys::Win32::Foundation::HANDLE);

#[cfg(windows)]
unsafe impl Send for InstanceMutex {}
#[cfg(windows)]
unsafe impl Sync for InstanceMutex {}

#[cfg(windows)]
impl Drop for InstanceMutex {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

/// Returns true when this process is the only running instance (and now
/// holds the guard), false when another instance already owns it.
#[cfg(windows)]
pub fn claim_single_instance() -> bool {
    use std::os::windows::ffi::OsStrExt;
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{
        GetLastError, ERROR_ALREADY_EXISTS, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::System::Threading::CreateMutexW;

    static INSTANCE_MUTEX: OnceLock<InstanceMutex> = OnceLock::new();
    let name: Vec<u16> = std::ffi::OsStr::new("Local\\WisprLocalAppInstance")
        .encode_wide()
        .chain(Some(0))
        .collect();
    let handle = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
    if handle.is_null() || handle == INVALID_HANDLE_VALUE {
        // Failure to create the guard should not make dictation unavailable.
        return true;
    }
    if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(handle);
        }
        return false;
    }
    INSTANCE_MUTEX.set(InstanceMutex(handle)).is_ok()
}

#[cfg(not(windows))]
pub fn claim_single_instance() -> bool {
    true
}
