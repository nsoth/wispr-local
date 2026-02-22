use rodio::{Decoder, OutputStream, Sink, Source};
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::{mpsc, Mutex};
use std::time::Duration;

enum SoundCommand {
    PlayStart,
    PlayStop,
    /// Update sound config at runtime
    UpdateConfig {
        start_sound: String,
        stop_sound: String,
        volume: f32,
    },
}

/// Persistent sound player with support for custom sound files.
pub struct SoundPlayer {
    sender: Mutex<mpsc::Sender<SoundCommand>>,
}

impl SoundPlayer {
    pub fn new(start_sound: String, stop_sound: String, volume: f32) -> Self {
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            let Ok((_stream, handle)) = OutputStream::try_default() else {
                log::error!("Failed to create audio output stream for sounds");
                return;
            };
            log::info!("Sound player initialized");

            let mut cfg_start = start_sound;
            let mut cfg_stop = stop_sound;
            let mut cfg_volume = volume;

            for cmd in rx {
                match cmd {
                    SoundCommand::UpdateConfig {
                        start_sound,
                        stop_sound,
                        volume,
                    } => {
                        cfg_start = start_sound;
                        cfg_stop = stop_sound;
                        cfg_volume = volume;
                        log::info!("Sound config updated (vol={:.0}%)", cfg_volume * 100.0);
                    }
                    SoundCommand::PlayStart => {
                        play_sound(&handle, &cfg_start, cfg_volume, true);
                    }
                    SoundCommand::PlayStop => {
                        play_sound(&handle, &cfg_stop, cfg_volume, false);
                    }
                }
            }
        });

        SoundPlayer {
            sender: Mutex::new(tx),
        }
    }

    pub fn play_start(&self) {
        if let Ok(tx) = self.sender.lock() {
            let _ = tx.send(SoundCommand::PlayStart);
        }
    }

    pub fn play_stop(&self) {
        if let Ok(tx) = self.sender.lock() {
            let _ = tx.send(SoundCommand::PlayStop);
        }
    }

    pub fn update_config(&self, start_sound: String, stop_sound: String, volume: f32) {
        if let Ok(tx) = self.sender.lock() {
            let _ = tx.send(SoundCommand::UpdateConfig {
                start_sound,
                stop_sound,
                volume,
            });
        }
    }
}

/// Play a sound: custom file if path is set, otherwise built-in tone.
fn play_sound(
    handle: &rodio::OutputStreamHandle,
    custom_path: &str,
    volume: f32,
    is_start: bool,
) {
    let Ok(sink) = Sink::try_new(handle) else {
        return;
    };
    sink.set_volume(volume);

    // Try custom file first
    if !custom_path.is_empty() {
        let path = PathBuf::from(custom_path);
        if path.exists() {
            match std::fs::File::open(&path) {
                Ok(file) => {
                    let reader = BufReader::new(file);
                    match Decoder::new(reader) {
                        Ok(source) => {
                            sink.append(source);
                            sink.sleep_until_end();
                            return;
                        }
                        Err(e) => log::warn!("Failed to decode {}: {}", custom_path, e),
                    }
                }
                Err(e) => log::warn!("Failed to open {}: {}", custom_path, e),
            }
        } else {
            log::warn!("Sound file not found: {}", custom_path);
        }
    }

    // Fallback: built-in tones (softer, more pleasant)
    if is_start {
        // Ascending soft chime: A4 → C#5 (major third, warm)
        let tone1 = rodio::source::SineWave::new(440.0)
            .take_duration(Duration::from_millis(60))
            .amplify(0.08)
            .fade_in(Duration::from_millis(10));
        let tone2 = rodio::source::SineWave::new(554.0)
            .take_duration(Duration::from_millis(80))
            .amplify(0.06)
            .fade_in(Duration::from_millis(10));
        sink.append(tone1);
        sink.append(tone2);
    } else {
        // Descending soft chime: C#5 → A4
        let tone1 = rodio::source::SineWave::new(554.0)
            .take_duration(Duration::from_millis(60))
            .amplify(0.08)
            .fade_in(Duration::from_millis(10));
        let tone2 = rodio::source::SineWave::new(440.0)
            .take_duration(Duration::from_millis(80))
            .amplify(0.06)
            .fade_in(Duration::from_millis(10));
        sink.append(tone1);
        sink.append(tone2);
    }
    sink.sleep_until_end();
}
