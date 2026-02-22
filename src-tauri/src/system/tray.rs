use tauri::{
    image::Image,
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Emitter, Manager,
};

pub fn setup_tray(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let start_item =
        MenuItem::with_id(app, "start_recording", "Start Recording", true, None::<&str>)?;
    let stop_item =
        MenuItem::with_id(app, "stop_recording", "Stop Recording", true, None::<&str>)?;
    let show_item =
        MenuItem::with_id(app, "show_window", "Show Window", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

    let menu = Menu::with_items(app, &[&start_item, &stop_item, &show_item, &quit_item])?;

    let icon = app
        .default_window_icon()
        .cloned()
        .unwrap_or_else(|| {
            // Fallback: generate a solid purple 32x32 icon
            let mut rgba = Vec::with_capacity(32 * 32 * 4);
            for _ in 0..(32 * 32) {
                rgba.extend_from_slice(&[124, 58, 237, 255]);
            }
            Image::new_owned(rgba, 32, 32)
        });

    let _tray = TrayIconBuilder::new()
        .icon(icon)
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

    Ok(())
}
