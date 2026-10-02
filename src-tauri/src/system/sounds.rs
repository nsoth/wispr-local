//! Start/stop chimes (plus "busy" and "cancel" cues).
//!
//! Every chime opens the *current* Windows default output device, plays, and
//! closes it again. A WASAPI stream is bound to the endpoint that was default
//! when it was created, so the previous design (one `OutputStream` opened at
//! app start) kept playing through the laptop speakers after the user switched
//! to headphones, and hung forever once that endpoint was invalidated. Opening
//! per play costs a few tens of milliseconds on a background thread and makes
//! both problems structurally impossible. The wait for playback is bounded,
//! so a dead render thread can never block the sound thread.

use cpal::traits::{DeviceTrait, HostTrait};
use rodio::buffer::SamplesBuffer;
use rodio::{Decoder, OutputStream, Sink, Source};
use std::io::BufReader;
use std::path::Path;
use std::sync::{mpsc, Mutex};
use std::time::{Duration, Instant};

/// Sample rate of the synthesized chimes.
pub const SAMPLE_RATE: u32 = 48_000;
/// Reference peak for every chime so the volume sliders mean the same thing for
/// built-in tones and user-supplied files (which used to play ~22 dB louder).
pub const CHIME_PEAK: f32 = 0.08;
const ATTACK_MS: u32 = 8;
const RELEASE_MS: u32 = 25;
/// Longest custom sound that is decoded; anything longer is truncated.
const MAX_CUSTOM_SECONDS: usize = 10;
/// Extra time allowed for a device to finish playing before the stream is
/// torn down (covers output buffering).
const WAIT_SLACK: Duration = Duration::from_millis(500);
/// Let the device drain its last buffer before the stream is closed.
const DRAIN_PAUSE: Duration = Duration::from_millis(40);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoundKind {
    Start,
    Stop,
    /// A press arrived while the previous recording is still being processed.
    Busy,
    /// The recording was discarded on purpose.
    Cancel,
}

impl SoundKind {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "start" => Some(Self::Start),
            "stop" => Some(Self::Stop),
            "busy" => Some(Self::Busy),
            "cancel" => Some(Self::Cancel),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SoundConfig {
    /// Path of a custom start sound, or empty for the built-in chime.
    pub start_sound: String,
    pub stop_sound: String,
    pub start_volume: f32,
    pub stop_volume: f32,
}

/// One note of a synthesized chime. `hz == 0.0` produces silence.
#[derive(Debug, Clone, Copy)]
pub struct Tone {
    pub hz: f32,
    pub ms: u32,
    pub peak: f32,
}

enum SoundCommand {
    Play {
        kind: SoundKind,
        volume_override: Option<f32>,
        reply: Option<mpsc::Sender<Result<String, String>>>,
    },
    UpdateConfig(SoundConfig),
}

/// Handle to the sound thread. Cheap to call from any thread; playback never
/// blocks the caller unless `play_and_wait` is used.
pub struct SoundPlayer {
    sender: Mutex<mpsc::Sender<SoundCommand>>,
}

impl SoundPlayer {
    pub fn new(config: SoundConfig) -> Self {
        let (tx, rx) = mpsc::channel();

        std::thread::Builder::new()
            .name("wispr-sounds".into())
            .spawn(move || {
                log::info!(
                    "Sound player ready (start {:.0}%, stop {:.0}%)",
                    config.start_volume * 100.0,
                    config.stop_volume * 100.0
                );
                let mut cfg = config;
                let mut last_device: Option<String> = None;
                for cmd in rx {
                    match cmd {
                        SoundCommand::UpdateConfig(new_cfg) => {
                            log::info!(
                                "Sound config updated (start {:.0}%, stop {:.0}%)",
                                new_cfg.start_volume * 100.0,
                                new_cfg.stop_volume * 100.0
                            );
                            cfg = new_cfg;
                        }
                        SoundCommand::Play {
                            kind,
                            volume_override,
                            reply,
                        } => {
                            let result = play_once(kind, &cfg, volume_override, &mut last_device);
                            if let Err(e) = &result {
                                log::warn!("Could not play {:?} chime: {e}", kind);
                            }
                            if let Some(tx) = reply {
                                let _ = tx.send(result);
                            }
                        }
                    }
                }
            })
            .expect("spawn sound thread");

        SoundPlayer {
            sender: Mutex::new(tx),
        }
    }

    pub fn play(&self, kind: SoundKind) {
        if let Ok(tx) = self.sender.lock() {
            let _ = tx.send(SoundCommand::Play {
                kind,
                volume_override: None,
                reply: None,
            });
        }
    }

    pub fn play_start(&self) {
        self.play(SoundKind::Start);
    }

    pub fn play_stop(&self) {
        self.play(SoundKind::Stop);
    }

    /// Play and wait for the result; returns the output device name. Used by
    /// the Settings "Test" buttons so the UI can show where the sound went.
    pub fn play_and_wait(
        &self,
        kind: SoundKind,
        volume_override: Option<f32>,
        timeout: Duration,
    ) -> Result<String, String> {
        let (reply_tx, reply_rx) = mpsc::channel();
        {
            let tx = self
                .sender
                .lock()
                .map_err(|_| "Sound player is unavailable".to_string())?;
            tx.send(SoundCommand::Play {
                kind,
                volume_override,
                reply: Some(reply_tx),
            })
            .map_err(|_| "Sound thread has stopped".to_string())?;
        }
        reply_rx
            .recv_timeout(timeout)
            .map_err(|_| "Sound playback timed out".to_string())?
    }

    pub fn update_config(&self, config: SoundConfig) {
        if let Ok(tx) = self.sender.lock() {
            let _ = tx.send(SoundCommand::UpdateConfig(config));
        }
    }
}

/// Open the current default output device, play one chime, close the device.
fn play_once(
    kind: SoundKind,
    cfg: &SoundConfig,
    volume_override: Option<f32>,
    last_device: &mut Option<String>,
) -> Result<String, String> {
    let (custom_path, volume) = match kind {
        SoundKind::Start => (cfg.start_sound.as_str(), cfg.start_volume),
        SoundKind::Stop => (cfg.stop_sound.as_str(), cfg.stop_volume),
        SoundKind::Busy | SoundKind::Cancel => ("", cfg.stop_volume),
    };
    let volume = volume_override.unwrap_or(volume).clamp(0.0, 1.0);

    let (samples, channels, rate) = load_custom(custom_path)
        .unwrap_or_else(|| (synth_chime(&builtin_tones(kind)), 1, SAMPLE_RATE));
    let frames = samples.len() / channels as usize;
    let expected = Duration::from_secs_f64(frames as f64 / rate as f64);

    let device = cpal::default_host()
        .default_output_device()
        .ok_or_else(|| "Windows reports no default output device".to_string())?;
    let name = device
        .name()
        .unwrap_or_else(|_| "unknown output device".to_string());
    let opened_at = Instant::now();
    let (stream, handle) = OutputStream::try_from_device(&device)
        .map_err(|e| format!("Could not open output device '{name}': {e}"))?;
    let open_ms = opened_at.elapsed().as_millis();
    if last_device.as_deref() != Some(name.as_str()) {
        log::info!("Sound output: {name} ({open_ms} ms to open)");
        *last_device = Some(name.clone());
    } else {
        log::debug!("Sound output: {name} ({open_ms} ms to open)");
    }

    let sink = Sink::try_new(&handle).map_err(|e| format!("Could not create audio sink: {e}"))?;
    sink.set_volume(volume);
    sink.append(SamplesBuffer::new(channels, rate, samples));

    let finished = wait_until_done(&sink, expected + WAIT_SLACK);
    if finished {
        std::thread::sleep(DRAIN_PAUSE);
    } else {
        log::warn!(
            "Sound output '{name}' did not finish a {} ms chime in time; closing the stream",
            expected.as_millis()
        );
    }
    drop(sink);
    drop(stream);
    Ok(name)
}

/// Poll the sink until it has played everything or the deadline passes.
/// Returns false on timeout (a device that was invalidated mid-play never
/// drains its queue; `Sink::sleep_until_end` would block forever there).
fn wait_until_done(sink: &Sink, deadline: Duration) -> bool {
    let started = Instant::now();
    while !sink.empty() {
        if started.elapsed() > deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    true
}

fn builtin_tones(kind: SoundKind) -> Vec<Tone> {
    match kind {
        // Ascending soft chime: A4 → C#5 (major third, warm).
        SoundKind::Start => vec![
            Tone {
                hz: 440.0,
                ms: 60,
                peak: CHIME_PEAK,
            },
            Tone {
                hz: 554.0,
                ms: 80,
                peak: 0.06,
            },
        ],
        // Descending soft chime: C#5 → A4.
        SoundKind::Stop => vec![
            Tone {
                hz: 554.0,
                ms: 60,
                peak: CHIME_PEAK,
            },
            Tone {
                hz: 440.0,
                ms: 80,
                peak: 0.06,
            },
        ],
        // Low double blip: "not now".
        SoundKind::Busy => vec![
            Tone {
                hz: 220.0,
                ms: 50,
                peak: 0.07,
            },
            Tone {
                hz: 0.0,
                ms: 40,
                peak: 0.0,
            },
            Tone {
                hz: 220.0,
                ms: 50,
                peak: 0.07,
            },
        ],
        // Single short low note.
        SoundKind::Cancel => vec![Tone {
            hz: 330.0,
            ms: 90,
            peak: 0.07,
        }],
    }
}

/// Render a sequence of sine notes at [`SAMPLE_RATE`], mono, each note shaped
/// by a raised-cosine attack/release so no note starts or ends with a click.
pub fn synth_chime(tones: &[Tone]) -> Vec<f32> {
    let rate = SAMPLE_RATE as f32;
    let mut out = Vec::new();
    for tone in tones {
        let n = SAMPLE_RATE as usize * tone.ms as usize / 1000;
        let attack = (SAMPLE_RATE as usize * ATTACK_MS as usize / 1000).min(n / 2);
        let release = (SAMPLE_RATE as usize * RELEASE_MS as usize / 1000).min(n / 2);
        for i in 0..n {
            let env = if i < attack {
                raised_cosine(i as f32 / attack as f32)
            } else if i >= n - release {
                raised_cosine((n - i) as f32 / release as f32)
            } else {
                1.0
            };
            let phase = 2.0 * std::f32::consts::PI * tone.hz * i as f32 / rate;
            out.push(phase.sin() * tone.peak * env);
        }
    }
    out
}

/// 0 → 0, 1 → 1, smooth in between.
fn raised_cosine(x: f32) -> f32 {
    0.5 - 0.5 * (std::f32::consts::PI * x.clamp(0.0, 1.0)).cos()
}

/// Scale samples so the loudest one sits at `target` (no-op for silence).
pub fn normalize_peak_to(samples: &mut [f32], target: f32) {
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if peak <= 0.0 {
        return;
    }
    let gain = target / peak;
    for s in samples.iter_mut() {
        *s *= gain;
    }
}

/// Decode a user-supplied sound file into (interleaved samples, channels,
/// sample rate), peak-normalized to [`CHIME_PEAK`] and capped at
/// [`MAX_CUSTOM_SECONDS`]. Returns None (with a log line) for an empty path, a
/// missing file or a decode failure so the built-in chime is used instead.
fn load_custom(path: &str) -> Option<(Vec<f32>, u16, u32)> {
    if path.is_empty() {
        return None;
    }
    let file_path = Path::new(path);
    if !file_path.exists() {
        log::warn!("Sound file not found: {path}");
        return None;
    }
    let file = match std::fs::File::open(file_path) {
        Ok(f) => f,
        Err(e) => {
            log::warn!("Failed to open {path}: {e}");
            return None;
        }
    };
    let decoder = match Decoder::new(BufReader::new(file)) {
        Ok(d) => d,
        Err(e) => {
            log::warn!("Failed to decode {path}: {e}");
            return None;
        }
    };
    let channels = decoder.channels().max(1);
    let rate = decoder.sample_rate().max(8_000);
    let max_samples = MAX_CUSTOM_SECONDS * rate as usize * channels as usize;
    let mut samples: Vec<f32> = decoder.convert_samples::<f32>().take(max_samples).collect();
    if samples.is_empty() {
        log::warn!("Sound file is empty: {path}");
        return None;
    }
    normalize_peak_to(&mut samples, CHIME_PEAK);
    fade_tail(&mut samples, channels as usize, rate);
    Some((samples, channels, rate))
}

/// Fade the last [`RELEASE_MS`] of a buffer so truncated files do not click.
fn fade_tail(samples: &mut [f32], channels: usize, rate: u32) {
    let frames = samples.len() / channels;
    let fade_frames = (rate as usize * RELEASE_MS as usize / 1000).min(frames);
    if fade_frames == 0 {
        return;
    }
    let first = frames - fade_frames;
    for frame in first..frames {
        let gain = raised_cosine((frames - frame) as f32 / fade_frames as f32);
        for ch in 0..channels {
            samples[frame * channels + ch] *= gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{normalize_peak_to, synth_chime, Tone, SAMPLE_RATE};

    #[test]
    fn chime_has_soft_edges_and_bounded_peak() {
        let samples = synth_chime(&[
            Tone {
                hz: 440.0,
                ms: 60,
                peak: 0.08,
            },
            Tone {
                hz: 554.0,
                ms: 80,
                peak: 0.06,
            },
        ]);
        assert_eq!(samples.len(), SAMPLE_RATE as usize * 140 / 1000);
        assert!(samples[0].abs() < 0.005, "attack starts from silence");
        assert!(
            samples[samples.len() - 1].abs() < 0.005,
            "release ends in silence, no click"
        );
        let peak = samples.iter().fold(0f32, |m, s| m.max(s.abs()));
        assert!(
            peak <= 0.08 * 1.001,
            "peak {peak} exceeds the configured 0.08"
        );
        assert!(peak > 0.05, "chime is audible");
        // The boundary between the two notes is also faded, so the loudest
        // sample right at the junction stays well below the note peak.
        let junction = SAMPLE_RATE as usize * 60 / 1000;
        assert!(samples[junction].abs() < 0.02);
        assert!(samples[junction - 1].abs() < 0.02);
    }

    #[test]
    fn custom_audio_is_normalized_to_reference_peak() {
        let mut samples = vec![0.0, 0.5, -1.0, 0.25];
        normalize_peak_to(&mut samples, 0.08);
        assert!((samples[2].abs() - 0.08).abs() < 1e-6);
        assert!((samples[1] - 0.04).abs() < 1e-6);

        let mut silent = vec![0.0; 4];
        normalize_peak_to(&mut silent, 0.08);
        assert!(silent.iter().all(|v| *v == 0.0), "silence stays silent");
    }

    #[test]
    fn busy_cue_contains_a_silent_gap() {
        let samples = synth_chime(&super::builtin_tones(super::SoundKind::Busy));
        let gap_start = SAMPLE_RATE as usize * 50 / 1000;
        let gap_end = SAMPLE_RATE as usize * 90 / 1000;
        assert!(samples[gap_start..gap_end].iter().all(|s| *s == 0.0));
    }
}
