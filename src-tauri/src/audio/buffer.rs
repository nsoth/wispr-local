use std::sync::{Arc, Mutex};

/// Simple thread-safe audio buffer that accumulates f32 samples at 16kHz.
/// Phase 1 uses a record-all-then-transcribe pattern.
#[derive(Clone)]
pub struct AudioBuffer {
    samples: Arc<Mutex<Vec<f32>>>,
}

impl AudioBuffer {
    pub fn new() -> Self {
        Self {
            // Pre-allocate for 30 seconds of 16kHz audio
            samples: Arc::new(Mutex::new(Vec::with_capacity(16000 * 30))),
        }
    }

    pub fn push_samples(&self, data: &[f32]) {
        if let Ok(mut buf) = self.samples.lock() {
            buf.extend_from_slice(data);
        }
    }

    pub fn take_samples(&self) -> Vec<f32> {
        if let Ok(mut buf) = self.samples.lock() {
            std::mem::take(&mut *buf)
        } else {
            Vec::new()
        }
    }

    pub fn clear(&self) {
        if let Ok(mut buf) = self.samples.lock() {
            buf.clear();
        }
    }

    pub fn len(&self) -> usize {
        self.samples.lock().map(|b| b.len()).unwrap_or(0)
    }

    /// Return a copy of the current samples without clearing the buffer.
    pub fn snapshot(&self) -> Vec<f32> {
        if let Ok(buf) = self.samples.lock() {
            buf.clone()
        } else {
            Vec::new()
        }
    }
}
