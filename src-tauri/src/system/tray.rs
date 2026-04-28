use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tauri::{
    image::Image,
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Emitter, Manager,
};

pub const TRAY_ID: &str = "main-tray";

/// Shared recording flag for the tray animator thread.
pub struct TrayAnimator {
    pub is_recording: Arc<AtomicBool>,
}

impl TrayAnimator {
    pub fn start(&self) {
        self.is_recording.store(true, Ordering::SeqCst);
    }

    pub fn stop(&self) {
        self.is_recording.store(false, Ordering::SeqCst);
    }
}

pub fn setup_tray(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let start_item =
        MenuItem::with_id(app, "start_recording", "Start Recording", true, None::<&str>)?;
    let stop_item =
        MenuItem::with_id(app, "stop_recording", "Stop Recording", true, None::<&str>)?;
    let show_item =
        MenuItem::with_id(app, "show_window", "Show Window", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

    let menu = Menu::with_items(app, &[&start_item, &stop_item, &show_item, &quit_item])?;

    let idle_icon = idle_icon(app);

    let _tray = TrayIconBuilder::with_id(TRAY_ID)
        .icon(idle_icon)
        .menu(&menu)
        .show_menu_on_left_click(false)
        .tooltip("Wispr Local - Idle")
        .on_menu_event(|app, event| match event.id.as_ref() {
            "start_recording" => {
                let _ = app.emit("tray-start-recording", ());
            }
            "stop_recording" => {
                let _ = app.emit("tray-stop-recording", ());
            }
            "show_window" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
            "quit" => {
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle();
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
        })
        .build(app)?;

    // Spawn animator thread. Cycles through pulsing red-dot frames while
    // recording; restores the idle icon once recording stops.
    let is_recording = Arc::new(AtomicBool::new(false));
    app.manage(TrayAnimator {
        is_recording: is_recording.clone(),
    });

    let app_handle = app.clone();
    std::thread::spawn(move || animator_loop(app_handle, is_recording));

    Ok(())
}

fn animator_loop(app: AppHandle, is_recording: Arc<AtomicBool>) {
    let frames = recording_frames();
    let idle = idle_icon(&app);
    let mut frame_idx = 0usize;
    let mut in_idle = true;

    loop {
        let recording = is_recording.load(Ordering::SeqCst);
        if let Some(tray) = app.tray_by_id(TRAY_ID) {
            if recording {
                let _ = tray.set_icon(Some(frames[frame_idx].clone()));
                let _ = tray.set_tooltip(Some("Wispr Local - Recording"));
                frame_idx = (frame_idx + 1) % frames.len();
                in_idle = false;
            } else if !in_idle {
                let _ = tray.set_icon(Some(idle.clone()));
                let _ = tray.set_tooltip(Some("Wispr Local - Idle"));
                frame_idx = 0;
                in_idle = true;
            }
        }
        let sleep_ms = if recording { 150 } else { 250 };
        std::thread::sleep(Duration::from_millis(sleep_ms));
    }
}

fn idle_icon(app: &AppHandle) -> Image<'static> {
    app.default_window_icon()
        .cloned()
        .map(|img| img.to_owned())
        .unwrap_or_else(|| {
            let mut rgba = Vec::with_capacity(32 * 32 * 4);
            for _ in 0..(32 * 32) {
                rgba.extend_from_slice(&[124, 58, 237, 255]);
            }
            Image::new_owned(rgba, 32, 32)
        })
}

/// Build pulsing red-dot frames for the recording indicator.
/// 32×32 RGBA, fully transparent background.
fn recording_frames() -> Vec<Image<'static>> {
    // Radius grows then shrinks — gives a visible pulse even at 16×16 scale.
    let radii = [7.0_f32, 9.0, 11.0, 9.0];
    radii.iter().map(|&r| red_dot_icon(r)).collect()
}

fn red_dot_icon(radius: f32) -> Image<'static> {
    const SIZE: u32 = 32;
    let cx = (SIZE as f32 - 1.0) / 2.0;
    let cy = (SIZE as f32 - 1.0) / 2.0;

    let mut rgba = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();

            // Anti-aliased edge over 1px.
            let alpha = if dist <= radius - 0.5 {
                1.0
            } else if dist >= radius + 0.5 {
                0.0
            } else {
                (radius + 0.5 - dist).clamp(0.0, 1.0)
            };

            let a = (alpha * 255.0) as u8;
            // Bright recording red (#ef4444-ish).
            rgba.extend_from_slice(&[239, 68, 68, a]);
        }
    }
    Image::new_owned(rgba, SIZE, SIZE)
}
