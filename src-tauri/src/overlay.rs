//! The floating recording pill: an opaque, always-on-top, non-activating
//! window clipped to a rounded region at the OS level and placed on the
//! monitor hosting the focused window.
//!
//! Three deliberate decisions live here (see the audit/memory notes before
//! changing them): the window is opaque + `SetWindowRgn`-clipped because
//! WebView2 transparency painted a square backdrop; the WebView2 controller
//! visibility is cycled on every show because the runtime stopped resuming
//! composition for this hidden→shown WS_EX_NOACTIVATE window; and placement
//! follows the focused window's monitor, recomputing the pill region at that
//! monitor's DPI.

use std::sync::Mutex;
use tauri::Manager;

use crate::settings::Settings;
use crate::state::{AppState, AppStatus};

/// Logical size of the overlay window; mirrors the overlay entry in
/// `tauri.conf.json`.
pub const LOGICAL_W: f64 = 312.0;
pub const LOGICAL_H: f64 = 52.0;

/// Clip the overlay window to a rounded pill (corner radius = height/2) via
/// SetWindowRgn, so nothing outside the pill exists at the compositor level.
/// WebView2 transparency proved unreliable here (square backdrop behind the
/// rounded CSS pill), so instead the window is opaque and simply has no pixels
/// outside the rounded region. `w`/`h` are physical pixels — the region must
/// be recomputed whenever the window lands on a monitor with a different DPI.
#[cfg(windows)]
pub fn apply_pill_region(overlay: &tauri::WebviewWindow, w: i32, h: i32) {
    let Ok(hwnd) = overlay.hwnd() else {
        log::warn!("Could not get overlay hwnd for pill region");
        return;
    };
    unsafe {
        use windows_sys::Win32::Graphics::Gdi::{CreateRoundRectRgn, DeleteObject, SetWindowRgn};
        // Ellipse w/h = window height -> fully rounded ends, matching the
        // CSS border-radius: 999px pill. The region takes ownership of rgn.
        let rgn = CreateRoundRectRgn(0, 0, w + 1, h + 1, h, h);
        if rgn.is_null() {
            log::warn!("Could not create overlay pill region");
        } else if SetWindowRgn(hwnd.0 as _, rgn, 1) == 0 {
            // Windows takes ownership only on success.
            let _ = DeleteObject(rgn as _);
            log::warn!("Could not apply overlay pill region");
        }
    }
}

#[cfg(not(windows))]
pub fn apply_pill_region(_overlay: &tauri::WebviewWindow, _w: i32, _h: i32) {}

/// Physical-pixel rect of the window that currently has keyboard focus — the
/// app the user is dictating into.
#[cfg(windows)]
fn foreground_window_rect() -> Option<windows_sys::Win32::Foundation::RECT> {
    use windows_sys::Win32::Foundation::RECT;
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowRect};
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.is_null() {
            return None;
        }
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        if GetWindowRect(hwnd, &mut rect) == 0 {
            return None;
        }
        Some(rect)
    }
}

/// The monitor hosting the focused window, falling back to the primary.
/// Dictation pastes into the focused app, so that monitor is where the user
/// is looking; pinning the overlay to the primary monitor made it invisible
/// whenever the user worked on another display.
fn target_monitor(overlay: &tauri::WebviewWindow) -> Option<tauri::Monitor> {
    #[cfg(windows)]
    if let Some(r) = foreground_window_rect() {
        let cx = r.left + (r.right - r.left) / 2;
        let cy = r.top + (r.bottom - r.top) / 2;
        if let Ok(monitors) = overlay.available_monitors() {
            let hit = monitors.into_iter().find(|m| {
                let p = m.position();
                let s = m.size();
                cx >= p.x && cx < p.x + s.width as i32 && cy >= p.y && cy < p.y + s.height as i32
            });
            if hit.is_some() {
                return hit;
            }
        }
    }
    overlay.primary_monitor().ok().flatten()
}

/// Position the overlay bottom-center on the monitor the user is working on
/// and clip it to the pill shape at that monitor's DPI.
pub fn place_overlay_window(app: &tauri::AppHandle) {
    let Some(overlay) = app.get_webview_window("overlay") else {
        return;
    };

    // Room for the default Windows taskbar (48 logical px) plus a small gap
    // so the overlay doesn't feel glued to it.
    const BOTTOM_MARGIN: f64 = 64.0;

    let Some(monitor) = target_monitor(&overlay) else {
        log::warn!("Overlay positioning skipped: no monitor found");
        return;
    };

    let scale = monitor.scale_factor();
    let mon_pos = monitor.position();
    let mon_size = monitor.size();

    // Physical size the window will have ON the target monitor. outer_size()
    // can't be used here: it reports the size at the window's current DPI,
    // which is stale while moving between monitors with different scales.
    let win_w = (LOGICAL_W * scale).round() as i32;
    let win_h = (LOGICAL_H * scale).round() as i32;
    let margin = (BOTTOM_MARGIN * scale).round() as i32;

    let x = mon_pos.x + (mon_size.width as i32 - win_w) / 2;
    let y = mon_pos.y + mon_size.height as i32 - win_h - margin;

    if let Err(e) = overlay.set_position(tauri::PhysicalPosition { x, y }) {
        log::warn!("Failed to position overlay: {}", e);
    }
    apply_pill_region(&overlay, win_w, win_h);
}

/// Sync the WebView2 controller's own visibility flag with the overlay
/// window. Since the late-July 2026 updates (WebView2 150.x runtime /
/// KB5101711), ShowWindow alone no longer resumes composition that was
/// suspended when this WS_EX_NOACTIVATE window was hidden: Win32 reports the
/// window visible with correct rect and region, but not a single pixel (not
/// even backgroundColor) reaches the screen. Dropping the controller to
/// hidden and back forces WebView2 to resume drawing; keeping it hidden
/// while the window is hidden also stops pointless background compositing.
#[cfg(windows)]
pub fn sync_overlay_webview_visibility(overlay: &tauri::WebviewWindow, visible: bool) {
    let result = overlay.with_webview(move |webview| unsafe {
        let controller = webview.controller();
        let _ = controller.SetIsVisible(false);
        if visible {
            if let Err(e) = controller.SetIsVisible(true) {
                log::error!("WebView2 SetIsVisible(true) failed: {e}");
            }
        }
    });
    if let Err(e) = result {
        log::error!("Overlay webview visibility sync failed: {e}");
    }
}

#[cfg(not(windows))]
pub fn sync_overlay_webview_visibility(_overlay: &tauri::WebviewWindow, _visible: bool) {}

/// Show the overlay (if the user has not disabled it), placed on the focused
/// window's monitor. Safe to call repeatedly.
pub fn show_overlay_if_enabled(app: &tauri::AppHandle) {
    let show = {
        let settings = app.state::<Mutex<Settings>>();
        let guard = settings.lock().unwrap();
        guard.show_overlay
    };
    if !show {
        return;
    }
    let Some(overlay) = app.get_webview_window("overlay") else {
        log::error!("Overlay window is gone — recording indicator unavailable");
        return;
    };
    place_overlay_window(app);
    if let Err(e) = overlay.show() {
        log::error!("Failed to show overlay: {}", e);
    }
    sync_overlay_webview_visibility(&overlay, true);
    // Crossing to a monitor with a different scale factor resizes the window
    // shortly after set_position; re-clip once the size has settled so the
    // pill isn't left with a stale region.
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        let recording = {
            let state = app.state::<Mutex<AppState>>();
            let s = state.lock().unwrap();
            s.status == AppStatus::Recording
        };
        if !recording {
            return;
        }
        if let Some(overlay) = app.get_webview_window("overlay") {
            if let Ok(size) = overlay.outer_size() {
                apply_pill_region(&overlay, size.width as i32, size.height as i32);
            }
        }
    });
}

pub fn hide_overlay(app: &tauri::AppHandle) {
    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.hide();
        sync_overlay_webview_visibility(&overlay, false);
    }
}
