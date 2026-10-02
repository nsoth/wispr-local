//! Microphone capture: opens the selected (or default) input device, converts
//! every packet to 16 kHz mono f32 and appends it to the shared buffer.

use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{
    FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig, SupportedStreamConfig,
};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

use super::buffer::AudioBuffer;
use super::devices;

/// Visual gain for the UI level meter only. Samples are stored raw; loudness
/// is peak-normalized right before transcription (see WhisperEngine), which
/// avoids the clipping distortion a fixed capture gain caused on loud speech.
const LEVEL_GAIN: f32 = 4.0;

/// Throttle level events to ~20Hz — enough for smooth waveform, cheap.
const LEVEL_INTERVAL: Duration = Duration::from_millis(50);

/// Wrapper to make cpal::Stream usable across threads.
/// On WASAPI (Windows), the stream handle is safe to move between threads.
/// The field is never read — it exists to keep the stream alive (dropping it
/// stops capture).
struct SendStream(#[allow(dead_code)] Stream);
unsafe impl Send for SendStream {}

pub struct AudioCapture {
    stream: Option<SendStream>,
    buffer: AudioBuffer,
    device_sample_rate: u32,
}

pub struct CaptureStart {
    pub sample_rate: u32,
    pub channels: u16,
    pub device_name: String,
    /// The preferred microphone was not used (absent or failed to open).
    pub used_fallback: bool,
    pub fallback_reason: Option<String>,
}

impl AudioCapture {
    pub fn new(buffer: AudioBuffer) -> Self {
        Self {
            stream: None,
            buffer,
            device_sample_rate: 48000,
        }
    }

    pub fn start(
        &mut self,
        app: Option<AppHandle>,
        preferred_device: Option<&str>,
    ) -> Result<CaptureStart, String> {
        // Never leave a previous stream running underneath a new one.
        self.stream = None;

        let selected = devices::select_input_device(preferred_device)?;
        let mut used_fallback = selected.used_fallback;
        let mut fallback_reason = selected.fallback_reason.clone();

        let (stream, config, device_name) =
            match self.open_stream(&selected.device, &selected.config, app.clone()) {
                Ok(stream) => (stream, selected.config.clone(), selected.name.clone()),
                Err(e) if !selected.used_fallback && preferred_device.is_some() => {
                    // The preferred microphone exists but refuses to open
                    // (seen with a virtual mic whose host app is not running):
                    // use the system default rather than failing the recording.
                    log::warn!(
                        "Preferred microphone '{}' failed to open ({e}); trying the system default",
                        selected.name
                    );
                    let fallback = devices::select_input_device(None)?;
                    if fallback.name == selected.name {
                        return Err(e);
                    }
                    let stream = self.open_stream(&fallback.device, &fallback.config, app)?;
                    used_fallback = true;
                    fallback_reason = Some(format!("'{}' could not be opened: {e}", selected.name));
                    (stream, fallback.config.clone(), fallback.name.clone())
                }
                Err(e) => return Err(e),
            };

        self.stream = Some(SendStream(stream));
        self.device_sample_rate = config.sample_rate().0;
        Ok(CaptureStart {
            sample_rate: self.device_sample_rate,
            channels: config.channels(),
            device_name,
            used_fallback,
            fallback_reason,
        })
    }

    fn open_stream(
        &mut self,
        device: &cpal::Device,
        supported: &SupportedStreamConfig,
        app: Option<AppHandle>,
    ) -> Result<Stream, String> {
        let sample_format = supported.sample_format();
        let config: StreamConfig = supported.clone().into();
        let channels = config.channels as usize;
        let native_rate = config.sample_rate.0;
        let buffer = self.buffer.clone();

        let stream = match sample_format {
            SampleFormat::F32 => {
                build_stream::<f32>(device, &config, buffer, app, channels, native_rate)?
            }
            SampleFormat::I16 => {
                build_stream::<i16>(device, &config, buffer, app, channels, native_rate)?
            }
            SampleFormat::U16 => {
                build_stream::<u16>(device, &config, buffer, app, channels, native_rate)?
            }
            SampleFormat::I32 => {
                build_stream::<i32>(device, &config, buffer, app, channels, native_rate)?
            }
            other => return Err(format!("Unsupported sample format: {other:?}")),
        };
        stream
            .play()
            .map_err(|e| format!("Failed to start stream: {e}"))?;
        Ok(stream)
    }

    pub fn stop(&mut self) {
        self.stream = None;
    }

    pub fn is_recording(&self) -> bool {
        self.stream.is_some()
    }

    pub fn device_sample_rate(&self) -> u32 {
        self.device_sample_rate
    }
}

/// Build the input stream for one sample type; every format converts to f32
/// before the shared mono/resample path.
fn build_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    buffer: AudioBuffer,
    app: Option<AppHandle>,
    channels: usize,
    native_rate: u32,
) -> Result<Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let app_cb = app.clone();
    let error_app = app;
    let mut last_emit = Instant::now() - LEVEL_INTERVAL;
    let mut limit_emitted = false;
    let mut error_emitted = false;
    let mut scratch: Vec<f32> = Vec::new();

    device
        .build_input_stream(
            config,
            move |data: &[T], _info: &cpal::InputCallbackInfo| {
                scratch.clear();
                scratch.extend(data.iter().map(|&s| f32::from_sample(s)));
                let mono = to_mono(&scratch, channels);
                let resampled = resample(&mono, native_rate, 16000);
                let limit_reached = buffer.push_samples(&resampled);

                if let Some(ref h) = app_cb {
                    if limit_reached && !limit_emitted {
                        limit_emitted = true;
                        let _ = h.emit(crate::events::RECORDING_LIMIT_REACHED, ());
                    }
                    let now = Instant::now();
                    if now.duration_since(last_emit) >= LEVEL_INTERVAL {
                        last_emit = now;
                        let _ = h.emit(crate::events::AUDIO_LEVEL, rms(&resampled) * LEVEL_GAIN);
                    }
                }
            },
            move |err| {
                log::error!("Audio stream error: {}", err);
                if !error_emitted {
                    error_emitted = true;
                    if let Some(ref h) = error_app {
                        let _ = h.emit(crate::events::AUDIO_STREAM_ERROR, err.to_string());
                    }
                }
            },
            None,
        )
        .map_err(|e| format!("Failed to build input stream: {e}"))
}

/// Convert multi-channel audio to mono by averaging channels.
fn to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    data.chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Root-mean-square amplitude in [0, 1] — used to drive the UI waveform.
fn rms(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = data.iter().map(|&s| s * s).sum();
    (sum_sq / data.len() as f32).sqrt()
}

/// Simple linear interpolation resampler (e.g., 48000 -> 16000 Hz).
fn resample(data: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate || data.is_empty() {
        return data.to_vec();
    }
    let ratio = source_rate as f64 / target_rate as f64;
    let output_len = (data.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let idx_floor = src_idx.floor() as usize;
        let idx_ceil = (idx_floor + 1).min(data.len() - 1);
        let frac = src_idx - idx_floor as f64;
        let sample = data[idx_floor] as f64 * (1.0 - frac) + data[idx_ceil] as f64 * frac;
        output.push(sample as f32);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{resample, rms, to_mono};

    #[test]
    fn mono_averages_channels() {
        assert_eq!(to_mono(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
        assert_eq!(to_mono(&[0.2, 0.4], 1), vec![0.2, 0.4]);
    }

    #[test]
    fn resample_keeps_duration_and_identity() {
        let input: Vec<f32> = (0..48_000).map(|i| (i % 7) as f32 / 7.0).collect();
        let out = resample(&input, 48_000, 16_000);
        assert_eq!(out.len(), 16_000);
        assert_eq!(resample(&input, 16_000, 16_000), input);
        assert!(resample(&[], 48_000, 16_000).is_empty());
    }

    #[test]
    fn rms_of_silence_and_full_scale() {
        assert_eq!(rms(&[]), 0.0);
        assert_eq!(rms(&[0.0, 0.0]), 0.0);
        assert!((rms(&[1.0, -1.0]) - 1.0).abs() < 1e-6);
    }
}
