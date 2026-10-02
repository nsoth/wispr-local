//! System tray: menu, click handling and the icon that mirrors the pipeline
//! phase (idle, pulsing red while recording, steady amber while the previous
//! recording is still being transcribed or pasted).

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tauri::{
    image::Image,
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Emitter, Manager,
};

pub const TRAY_ID: &str = "main-tray";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TrayPhase {
    Idle = 0,
    Recording = 1,
    Processing = 2,
}

impl TrayPhase {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => TrayPhase::Recording,
            2 => TrayPhase::Processing,
            _ => TrayPhase::Idle,
        }
    }

    fn tooltip(self) -> &'static str {
        match self {
            TrayPhase::Idle => "Wispr Local - Idle",
            TrayPhase::Recording => "Wispr Local - Recording",
            TrayPhase::Processing => "Wispr Local - Transcribing",
        }
    }
}

/// Shared phase for the tray animator thread.
pub struct TrayAnimator {
    phase: Arc<AtomicU8>,
}

impl TrayAnimator {
    pub fn set_phase(&self, phase: TrayPhase) {
        self.phase.store(phase as u8, Ordering::SeqCst);
    }

    pub fn phase(&self) -> TrayPhase {
        TrayPhase::from_u8(self.phase.load(Ordering::SeqCst))
    }
}

pub fn setup_tray(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let start_item = MenuItem::with_id(
        app,
        "start_recording",
        "Start Recording",
        true,
        None::<&str>,
    )?;
    let stop_item = MenuItem::with_id(app, "stop_recording", "Stop Recording", true, None::<&str>)?;
    let show_item = MenuItem::with_id(app, "show_window", "Show Window", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

    let menu = Menu::with_items(app, &[&start_item, &stop_item, &show_item, &quit_item])?;

    let idle_icon = idle_icon(app);

    let _tray = TrayIconBuilder::with_id(TRAY_ID)
        .icon(idle_icon)
        .menu(&menu)
        .show_menu_on_left_click(false)
        .tooltip(TrayPhase::Idle.tooltip())
        .on_menu_event(|app, event| match event.id.as_ref() {
            "start_recording" => {
                let _ = app.emit(crate::events::REQUEST_START_RECORDING, ());
            }
            "stop_recording" => {
                let _ = app.emit(crate::events::REQUEST_STOP_RECORDING, ());
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
    // recording, shows a steady amber dot while processing, restores the idle
    // icon otherwise.
    let phase = Arc::new(AtomicU8::new(TrayPhase::Idle as u8));
    app.manage(TrayAnimator {
        phase: phase.clone(),
    });

    let app_handle = app.clone();
    std::thread::Builder::new()
        .name("wispr-tray".into())
        .spawn(move || animator_loop(app_handle, phase))?;

    Ok(())
}

fn animator_loop(app: AppHandle, phase: Arc<AtomicU8>) {
    let frames = recording_frames();
    let idle = idle_icon(&app);
    let processing = dot_icon(9.0, [245, 158, 11]);
    let mut frame_idx = 0usize;
    let mut shown = TrayPhase::Idle;
    let mut first = true;

    loop {
        let current = TrayPhase::from_u8(phase.load(Ordering::SeqCst));
        if let Some(tray) = app.tray_by_id(TRAY_ID) {
            match current {
                TrayPhase::Recording => {
                    let _ = tray.set_icon(Some(frames[frame_idx].clone()));
                    frame_idx = (frame_idx + 1) % frames.len();
                }
                TrayPhase::Processing => {
                    if shown != TrayPhase::Processing || first {
                        let _ = tray.set_icon(Some(processing.clone()));
                    }
                }
                TrayPhase::Idle => {
                    if shown != TrayPhase::Idle {
                        let _ = tray.set_icon(Some(idle.clone()));
                        frame_idx = 0;
                    }
                }
            }
            if shown != current || first {
                let _ = tray.set_tooltip(Some(current.tooltip()));
            }
        }
        shown = current;
        first = false;
        let sleep_ms = if current == TrayPhase::Recording {
            150
        } else {
            250
        };
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
    radii.iter().map(|&r| dot_icon(r, [239, 68, 68])).collect()
}

fn dot_icon(radius: f32, rgb: [u8; 3]) -> Image<'static> {
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
            rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], a]);
        }
    }
    Image::new_owned(rgba, SIZE, SIZE)
}

#[cfg(test)]
mod tests {
    use super::TrayPhase;

    #[test]
    fn phases_round_trip_through_the_atomic() {
        for phase in [TrayPhase::Idle, TrayPhase::Recording, TrayPhase::Processing] {
            assert_eq!(TrayPhase::from_u8(phase as u8), phase);
        }
        assert_eq!(TrayPhase::from_u8(42), TrayPhase::Idle);
    }
}
