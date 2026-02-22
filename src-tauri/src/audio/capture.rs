use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};

use super::buffer::AudioBuffer;

/// Microphone gain multiplier. Boost quiet mics for better recognition.
const MIC_GAIN: f32 = 4.0;

/// Wrapper to make cpal::Stream usable across threads.
/// On WASAPI (Windows), the stream handle is safe to move between threads.
struct SendStream(Stream);
unsafe impl Send for SendStream {}

pub struct AudioCapture {
    stream: Option<SendStream>,
    buffer: AudioBuffer,
    device_sample_rate: u32,
}

// AudioCapture is Send+Sync because SendStream is Send and other fields are Send+Sync
unsafe impl Sync for AudioCapture {}

impl AudioCapture {
    pub fn new(buffer: AudioBuffer) -> Self {
        Self {
            stream: None,
            buffer,
            device_sample_rate: 48000,
        }
    }

    pub fn start(&mut self) -> Result<u32, String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("No input device found")?;

        let supported_config = device
            .default_input_config()
            .map_err(|e| format!("Failed to get default input config: {}", e))?;

        let sample_format = supported_config.sample_format();
        let config: StreamConfig = supported_config.into();
        self.device_sample_rate = config.sample_rate.0;
        let channels = config.channels as usize;
        let native_rate = self.device_sample_rate;

        let buffer = self.buffer.clone();

        let stream = match sample_format {
            SampleFormat::F32 => device
                .build_input_stream(
                    &config,
                    move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                        let mono = to_mono(data, channels);
                        let resampled = resample(&mono, native_rate, 16000);
                        let amplified = apply_gain(&resampled, MIC_GAIN);
                        buffer.push_samples(&amplified);
                    },
                    |err| log::error!("Audio stream error: {}", err),
                    None,
                )
                .map_err(|e| format!("Failed to build f32 input stream: {}", e))?,
            SampleFormat::I16 => device
                .build_input_stream(
                    &config,
                    move |data: &[i16], _info: &cpal::InputCallbackInfo| {
                        let float_data: Vec<f32> =
                            data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                        let mono = to_mono(&float_data, channels);
                        let resampled = resample(&mono, native_rate, 16000);
                        let amplified = apply_gain(&resampled, MIC_GAIN);
                        buffer.push_samples(&amplified);
                    },
                    |err| log::error!("Audio stream error: {}", err),
                    None,
                )
                .map_err(|e| format!("Failed to build i16 input stream: {}", e))?,
            _ => return Err(format!("Unsupported sample format: {:?}", sample_format)),
        };

        stream
            .play()
            .map_err(|e| format!("Failed to start stream: {}", e))?;
        self.stream = Some(SendStream(stream));
        Ok(self.device_sample_rate)
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

/// Convert multi-channel audio to mono by averaging channels.
fn to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    if channels == 1 {
        return data.to_vec();
    }
    data.chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Apply gain and clamp to [-1.0, 1.0] to avoid clipping.
fn apply_gain(data: &[f32], gain: f32) -> Vec<f32> {
    data.iter().map(|&s| (s * gain).clamp(-1.0, 1.0)).collect()
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
