//! The shared sample buffer: the capture callback appends 16 kHz mono f32
//! samples, the preview and final passes take snapshots. Bounded at 30 min.

use std::sync::{Arc, Mutex};

/// Keep recording memory bounded even when hands-free mode is left running.
/// At 16 kHz mono f32, 30 minutes is roughly 110 MiB.
pub const MAX_RECORDING_SAMPLES: usize = 16_000 * 60 * 30;
/// Two minutes pre-allocated (7.7 MB): the owner's recordings are mostly
/// under a minute, so the real-time capture callback never has to grow the
/// vector mid-dictation; longer hands-free sessions still double as needed.
const INITIAL_CAPACITY: usize = 16_000 * 120;

/// Thread-safe audio buffer that accumulates f32 samples at 16kHz.
#[derive(Clone)]
pub struct AudioBuffer {
    samples: Arc<Mutex<Vec<f32>>>,
}

impl Default for AudioBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioBuffer {
    pub fn new() -> Self {
        Self {
            // Pre-allocate for 30 seconds of 16kHz audio
            samples: Arc::new(Mutex::new(Vec::with_capacity(INITIAL_CAPACITY))),
        }
    }

    /// Append samples and return true once the recording limit has been hit.
    pub fn push_samples(&self, data: &[f32]) -> bool {
        self.push_samples_with_limit(data, MAX_RECORDING_SAMPLES)
    }

    fn push_samples_with_limit(&self, data: &[f32], limit: usize) -> bool {
        if let Ok(mut buf) = self.samples.lock() {
            let remaining = limit.saturating_sub(buf.len());
            if remaining > 0 {
                buf.extend_from_slice(&data[..data.len().min(remaining)]);
            }
            buf.len() >= limit
        } else {
            false
        }
    }

    pub fn take_samples(&self) -> Vec<f32> {
        if let Ok(mut buf) = self.samples.lock() {
            // Retain a useful allocation for the next recording instead of
            // forcing the real-time audio callback to grow from zero again.
            let mut replacement = Vec::with_capacity(INITIAL_CAPACITY);
            std::mem::swap(&mut *buf, &mut replacement);
            replacement
        } else {
            Vec::new()
        }
    }

    pub fn clear(&self) {
        if let Ok(mut buf) = self.samples.lock() {
            buf.clear();
        }
    }

    /// Samples captured so far.
    pub fn len(&self) -> usize {
        self.samples.lock().map(|b| b.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy everything captured after `offset` (for the crash spool).
    pub fn snapshot_from(&self, offset: usize) -> Vec<f32> {
        if let Ok(buf) = self.samples.lock() {
            if offset >= buf.len() {
                return Vec::new();
            }
            buf[offset..].to_vec()
        } else {
            Vec::new()
        }
    }

    /// Copy at most the newest `max_samples` without cloning a long recording.
    pub fn snapshot_tail(&self, max_samples: usize) -> Vec<f32> {
        if let Ok(buf) = self.samples.lock() {
            let start = buf.len().saturating_sub(max_samples);
            buf[start..].to_vec()
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AudioBuffer;

    #[test]
    fn snapshot_tail_only_copies_requested_samples() {
        let buffer = AudioBuffer::new();
        assert!(!buffer.push_samples(&[1.0, 2.0, 3.0, 4.0]));
        assert_eq!(buffer.snapshot_tail(2), vec![3.0, 4.0]);
        assert_eq!(buffer.snapshot_tail(10), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn recording_buffer_is_bounded() {
        let buffer = AudioBuffer::new();
        assert!(buffer.push_samples_with_limit(&[1.0, 2.0, 3.0, 4.0], 3));
        assert_eq!(buffer.take_samples(), vec![1.0, 2.0, 3.0]);
    }
}
