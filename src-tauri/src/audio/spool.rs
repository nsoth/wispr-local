//! Crash insurance for the recording in progress.
//!
//! whisper.cpp aborts the whole process on a CUDA error, and the only copy of
//! the audio used to live in memory. While recording, a writer thread appends
//! the new 16 kHz samples to `pending.pcm` (raw little-endian i16, ~1.9 MB per
//! minute) about once a second; the file is deleted once the dictation
//! finished normally. If the app starts and finds a spool file with at least
//! half a second of audio, that recording is transcribed and put into the
//! history and the clipboard instead of being lost.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::buffer::AudioBuffer;

pub const SPOOL_FILE: &str = "pending.pcm";
const FLUSH_INTERVAL: Duration = Duration::from_millis(1000);
/// Shorter spools are accidental taps, not recordings worth recovering.
pub const MIN_RECOVER_SAMPLES: usize = 16_000 / 2;

pub fn spool_path(data_dir: &Path) -> PathBuf {
    data_dir.join(SPOOL_FILE)
}

/// Handle to the writer thread; `stop` flushes the tail and joins.
pub struct SpoolWriter {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SpoolWriter {
    /// Start spooling `buffer` into `path` (truncating any previous file).
    pub fn start(path: PathBuf, buffer: AudioBuffer) -> Result<SpoolWriter, String> {
        let file = std::fs::File::create(&path)
            .map_err(|e| format!("Could not create the recording spool: {e}"))?;
        let stop = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("wispr-spool".into())
            .spawn(move || writer_loop(file, buffer, stop_flag))
            .map_err(|e| format!("Could not start the spool thread: {e}"))?;
        Ok(SpoolWriter {
            stop,
            thread: Some(thread),
        })
    }

    /// Flush everything captured so far and stop the thread.
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for SpoolWriter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn writer_loop(mut file: std::fs::File, buffer: AudioBuffer, stop: Arc<AtomicBool>) {
    use std::io::Write;
    let mut written = 0usize;
    let mut bytes = Vec::new();
    loop {
        let stopping = stop.load(Ordering::Relaxed);
        let fresh = buffer.snapshot_from(written);
        if !fresh.is_empty() {
            bytes.clear();
            bytes.reserve(fresh.len() * 2);
            for s in &fresh {
                bytes.extend_from_slice(&to_i16(*s).to_le_bytes());
            }
            if let Err(e) = file.write_all(&bytes) {
                log::warn!("Recording spool write failed: {e}");
                return;
            }
            written += fresh.len();
        }
        if stopping {
            let _ = file.sync_data();
            return;
        }
        std::thread::sleep(FLUSH_INTERVAL);
    }
}

fn to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
}

/// Read a spool file back into 16 kHz f32 samples.
pub fn read_spool(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("Could not read the spool: {e}"))?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / i16::MAX as f32)
        .collect())
}

/// Spool left by a crashed or killed run, if it holds enough audio.
pub fn pending_recovery(data_dir: &Path) -> Option<PathBuf> {
    let path = spool_path(data_dir);
    let len = std::fs::metadata(&path).map(|m| m.len()).ok()?;
    if (len as usize) / 2 >= MIN_RECOVER_SAMPLES {
        Some(path)
    } else {
        let _ = std::fs::remove_file(&path);
        None
    }
}

pub fn remove_spool(data_dir: &Path) {
    let path = spool_path(data_dir);
    match std::fs::remove_file(&path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => log::warn!("Could not remove the recording spool: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{pending_recovery, read_spool, spool_path, SpoolWriter, MIN_RECOVER_SAMPLES};
    use crate::audio::buffer::AudioBuffer;
    use crate::config::test_dir;

    #[test]
    fn spool_round_trips_the_captured_audio() {
        let dir = test_dir("spool");
        let buffer = AudioBuffer::new();
        let samples: Vec<f32> = (0..20_000)
            .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
            .collect();
        buffer.push_samples(&samples[..8_000]);
        let writer = SpoolWriter::start(spool_path(&dir), buffer.clone()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        buffer.push_samples(&samples[8_000..]);
        writer.stop();

        let back = read_spool(&spool_path(&dir)).unwrap();
        assert_eq!(back.len(), samples.len());
        let max_err = back
            .iter()
            .zip(&samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "i16 quantization error {max_err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn short_spools_are_discarded_and_long_ones_offered() {
        let dir = test_dir("spool-recover");
        let path = spool_path(&dir);
        std::fs::write(&path, vec![0u8; (MIN_RECOVER_SAMPLES - 1) * 2]).unwrap();
        assert!(pending_recovery(&dir).is_none());
        assert!(!path.exists(), "too-short spool removed");
        std::fs::write(&path, vec![0u8; MIN_RECOVER_SAMPLES * 2]).unwrap();
        assert_eq!(pending_recovery(&dir), Some(path));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
