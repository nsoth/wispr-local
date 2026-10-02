//! System tray: a menu that mirrors the app state (start/stop, cancel,
//! language, AI formatting, paused hotkey, restart on GPU), click handling
//! and an icon that mirrors the pipeline phase (idle, pulsing red while
//! recording, steady amber while the previous recording is still being
//! transcribed or pasted).

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tauri::{
    image::Image,
    menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem, Submenu},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Emitter, Manager, Wry,
};

use crate::events;
use crate::settings::Settings;
use crate::state::{lock_or_recover, AppState, AppStatus};
use crate::transcription::engine::LanguageMode;

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

/// Handles to the menu items whose text / enabled / checked state follows
/// the app state (see [`refresh`]).
pub struct TrayMenu {
    status: MenuItem<Wry>,
    start_stop: MenuItem<Wry>,
    cancel: MenuItem<Wry>,
    lang_auto: CheckMenuItem<Wry>,
    lang_ru: CheckMenuItem<Wry>,
    lang_en: CheckMenuItem<Wry>,
    ai_enabled: CheckMenuItem<Wry>,
    hotkey_paused: CheckMenuItem<Wry>,
    restart_gpu: MenuItem<Wry>,
    copy_last: MenuItem<Wry>,
}

/// What the state-dependent items should show. Pure, for the tests.
pub fn tray_labels(status: &AppStatus, hotkey: &str) -> (String, &'static str, bool, bool) {
    let status_text = match status {
        AppStatus::Idle => "Idle".to_string(),
        AppStatus::Recording => "Recording…".to_string(),
        AppStatus::Transcribing => "Transcribing…".to_string(),
        AppStatus::Formatting => "Formatting…".to_string(),
        AppStatus::Injecting => "Pasting…".to_string(),
        AppStatus::Error { message, .. } => format!("Error: {message}"),
    };
    let (start_stop, start_enabled) = match status {
        AppStatus::Recording => ("Stop and paste", true),
        AppStatus::Idle | AppStatus::Error { .. } => ("Start hands-free recording", true),
        _ => ("Start hands-free recording", false),
    };
    let cancel_enabled = matches!(
        status,
        AppStatus::Recording | AppStatus::Transcribing | AppStatus::Formatting
    );
    let status_line = if hotkey.is_empty() {
        status_text
    } else {
        format!("{status_text}  ·  {hotkey}")
    };
    (status_line, start_stop, start_enabled, cancel_enabled)
}

pub fn setup_tray(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let status = MenuItem::with_id(app, "status", "Idle", false, None::<&str>)?;
    let start_stop = MenuItem::with_id(
        app,
        "start_stop",
        "Start hands-free recording",
        true,
        None::<&str>,
    )?;
    let cancel = MenuItem::with_id(app, "cancel", "Cancel recording", false, None::<&str>)?;
    let copy_last = MenuItem::with_id(
        app,
        "copy_last",
        "Copy last transcript",
        false,
        None::<&str>,
    )?;

    let lang_auto = CheckMenuItem::with_id(
        app,
        "lang_auto",
        "Auto (Russian / English)",
        true,
        true,
        None::<&str>,
    )?;
    let lang_ru = CheckMenuItem::with_id(app, "lang_ru", "Russian", true, false, None::<&str>)?;
    let lang_en = CheckMenuItem::with_id(app, "lang_en", "English", true, false, None::<&str>)?;
    let language = Submenu::with_items(app, "Language", true, &[&lang_auto, &lang_ru, &lang_en])?;

    let ai_enabled = CheckMenuItem::with_id(
        app,
        "ai_enabled",
        "AI formatting",
        true,
        false,
        None::<&str>,
    )?;
    let hotkey_paused = CheckMenuItem::with_id(
        app,
        "hotkey_paused",
        "Pause hotkey",
        true,
        false,
        None::<&str>,
    )?;
    let restart_gpu = MenuItem::with_id(app, "restart_gpu", "Restart on GPU", false, None::<&str>)?;

    let show_item = MenuItem::with_id(app, "show_window", "Show window", true, None::<&str>)?;
    let settings_item = MenuItem::with_id(app, "open_settings", "Settings…", true, None::<&str>)?;
    let log_item = MenuItem::with_id(app, "open_log", "Open log folder", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

    let menu = Menu::with_items(
        app,
        &[
            &status,
            &PredefinedMenuItem::separator(app)?,
            &start_stop,
            &cancel,
            &copy_last,
            &PredefinedMenuItem::separator(app)?,
            &language,
            &ai_enabled,
            &hotkey_paused,
            &restart_gpu,
            &PredefinedMenuItem::separator(app)?,
            &show_item,
            &settings_item,
            &log_item,
            &quit_item,
        ],
    )?;

    app.manage(TrayMenu {
        status,
        start_stop,
        cancel,
        lang_auto,
        lang_ru,
        lang_en,
        ai_enabled,
        hotkey_paused,
        restart_gpu,
        copy_last,
    });

    let idle_icon = idle_icon(app);

    let _tray = TrayIconBuilder::with_id(TRAY_ID)
        .icon(idle_icon)
        .menu(&menu)
        .show_menu_on_left_click(false)
        .tooltip(TrayPhase::Idle.tooltip())
        .on_menu_event(|app, event| on_menu(app, event.id.as_ref()))
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main_window(tray.app_handle());
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

    refresh(app);
    Ok(())
}

fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

fn on_menu(app: &AppHandle, id: &str) {
    match id {
        "start_stop" => {
            let recording = {
                let state = app.state::<Mutex<AppState>>();
                let s = lock_or_recover(&state);
                s.status == AppStatus::Recording
            };
            if recording {
                let _ = app.emit(events::REQUEST_STOP_RECORDING, ());
            } else {
                let _ = app.emit(events::REQUEST_START_HANDS_FREE, ());
            }
        }
        "cancel" => {
            let _ = app.emit(events::REQUEST_CANCEL_RECORDING, ());
        }
        "copy_last" => {
            let text = {
                let state = app.state::<Mutex<AppState>>();
                let s = lock_or_recover(&state);
                s.last_transcription.clone()
            };
            if !text.is_empty() {
                match crate::system::text_injection::copy_only(&text) {
                    Ok(()) => log::info!("Last transcript copied from the tray"),
                    Err(e) => log::warn!("Could not copy the last transcript: {e}"),
                }
            }
        }
        "lang_auto" => set_language_from_tray(app, LanguageMode::Auto),
        "lang_ru" => set_language_from_tray(app, LanguageMode::Russian),
        "lang_en" => set_language_from_tray(app, LanguageMode::English),
        "ai_enabled" => {
            let enabled = {
                let settings = app.state::<Mutex<Settings>>();
                let s = lock_or_recover(&settings);
                s.ai.enabled
            };
            if let Err(e) = crate::commands::apply_ai_enabled(app, !enabled) {
                log::warn!("Could not toggle AI formatting from the tray: {e}");
            }
        }
        "hotkey_paused" => {
            let paused = {
                let state = app.state::<Mutex<AppState>>();
                let mut s = lock_or_recover(&state);
                s.hotkey_paused = !s.hotkey_paused;
                s.hotkey_paused
            };
            log::info!(
                "Hotkey {} from the tray",
                if paused { "paused" } else { "resumed" }
            );
            refresh(app);
        }
        "restart_gpu" => {
            log::info!("Restart on GPU requested from the tray");
            app.exit(crate::supervisor::RESTART_ON_GPU_CODE);
        }
        "show_window" => show_main_window(app),
        "open_settings" => {
            show_main_window(app);
            let _ = app.emit(events::OPEN_SETTINGS, ());
        }
        "open_log" => {
            let log_path = app
                .state::<crate::config::AppConfig>()
                .data_dir
                .join("wispr.log");
            #[cfg(windows)]
            {
                let _ = std::process::Command::new("explorer.exe")
                    .arg(format!("/select,{}", log_path.display()))
                    .spawn();
            }
            #[cfg(not(windows))]
            {
                log::info!("Log file: {}", log_path.display());
            }
        }
        "quit" => quit_when_idle(app.clone()),
        _ => {}
    }
}

fn set_language_from_tray(app: &AppHandle, language: LanguageMode) {
    if let Err(e) = crate::commands::apply_language(app, language) {
        log::warn!("Could not change the language from the tray: {e}");
    }
}

/// Quit, but give an in-flight transcription/paste up to two seconds to
/// finish so the clipboard is restored and the text is not lost mid-paste.
fn quit_when_idle(app: AppHandle) {
    std::thread::spawn(move || {
        for _ in 0..20 {
            let busy = {
                let state = app.state::<Mutex<AppState>>();
                let s = lock_or_recover(&state);
                matches!(
                    s.status,
                    AppStatus::Transcribing | AppStatus::Formatting | AppStatus::Injecting
                )
            };
            if !busy {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        log::info!("Quit requested from the tray");
        app.exit(0);
    });
}

/// Bring every state-dependent item in line with AppState and Settings.
/// Cheap; called on each status / model / settings change.
pub fn refresh(app: &AppHandle) {
    let Some(menu) = app.try_state::<TrayMenu>() else {
        return;
    };
    let (status, backend_cpu, paused, has_last) = {
        let state = app.state::<Mutex<AppState>>();
        let s = lock_or_recover(&state);
        (
            s.status.clone(),
            s.model.backend() == "CPU",
            s.hotkey_paused,
            !s.last_transcription.is_empty(),
        )
    };
    let (language, ai_on, ai_configured, hotkey) = {
        let settings = app.state::<Mutex<Settings>>();
        let s = lock_or_recover(&settings);
        (
            s.language,
            s.ai.enabled,
            s.ai.provider != crate::formatting::AiProvider::None,
            s.hotkey.clone(),
        )
    };
    let (status_line, start_stop, start_enabled, cancel_enabled) = tray_labels(&status, &hotkey);
    let _ = menu.status.set_text(status_line);
    let _ = menu.start_stop.set_text(start_stop);
    let _ = menu.start_stop.set_enabled(start_enabled);
    let _ = menu.cancel.set_enabled(cancel_enabled);
    let _ = menu.copy_last.set_enabled(has_last);
    let _ = menu.lang_auto.set_checked(matches!(
        language,
        LanguageMode::Auto | LanguageMode::Unknown
    ));
    let _ = menu.lang_ru.set_checked(language == LanguageMode::Russian);
    let _ = menu.lang_en.set_checked(language == LanguageMode::English);
    let _ = menu.ai_enabled.set_checked(ai_on && ai_configured);
    let _ = menu.ai_enabled.set_enabled(ai_configured);
    let _ = menu.hotkey_paused.set_checked(paused);
    let _ = menu.restart_gpu.set_enabled(backend_cpu);
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
    use super::{tray_labels, TrayPhase};
    use crate::state::AppStatus;

    #[test]
    fn phases_round_trip_through_the_atomic() {
        for phase in [TrayPhase::Idle, TrayPhase::Recording, TrayPhase::Processing] {
            assert_eq!(TrayPhase::from_u8(phase as u8), phase);
        }
        assert_eq!(TrayPhase::from_u8(42), TrayPhase::Idle);
    }

    #[test]
    fn menu_labels_follow_the_status() {
        let (status, start_stop, enabled, cancel) =
            tray_labels(&AppStatus::Idle, "Ctrl+Shift+Space");
        assert_eq!(status, "Idle  ·  Ctrl+Shift+Space");
        assert_eq!(start_stop, "Start hands-free recording");
        assert!(enabled);
        assert!(!cancel);

        let (status, start_stop, enabled, cancel) = tray_labels(&AppStatus::Recording, "");
        assert_eq!(status, "Recording…");
        assert_eq!(start_stop, "Stop and paste");
        assert!(enabled);
        assert!(cancel);

        let (_, _, enabled, cancel) = tray_labels(&AppStatus::Transcribing, "");
        assert!(!enabled, "cannot start while transcribing");
        assert!(cancel, "but can cancel the paste");

        let (_, _, enabled, cancel) = tray_labels(&AppStatus::Injecting, "");
        assert!(!enabled);
        assert!(!cancel, "too late to cancel a paste in flight");
    }
}
