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

/// Folder that keeps finished recordings so a bad transcription can always be
/// redone from the audio.
pub const RECORDINGS_DIR: &str = "recordings";

pub fn recordings_dir(data_dir: &Path) -> PathBuf {
    data_dir.join(RECORDINGS_DIR)
}

/// 44-byte RIFF header for 16 kHz mono 16-bit PCM.
fn wav_header(data_len: u32) -> [u8; 44] {
    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&(36 + data_len).to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM
    h[22..24].copy_from_slice(&1u16.to_le_bytes()); // mono
    h[24..28].copy_from_slice(&16_000u32.to_le_bytes());
    h[28..32].copy_from_slice(&32_000u32.to_le_bytes()); // byte rate
    h[32..34].copy_from_slice(&2u16.to_le_bytes()); // block align
    h[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_len.to_le_bytes());
    h
}

/// Move the finished spool into the recordings folder as a WAV named after
/// the time and the outcome. `None` when there is no spool (or too little).
pub fn archive_spool(data_dir: &Path, label: &str) -> Option<PathBuf> {
    let spool = spool_path(data_dir);
    let bytes = std::fs::read(&spool).ok()?;
    if bytes.len() / 2 < MIN_RECOVER_SAMPLES {
        let _ = std::fs::remove_file(&spool);
        return None;
    }
    let dir = recordings_dir(data_dir);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::warn!("Could not create the recordings folder: {e}");
        return None;
    }
    let stamp = crate::config::timestamp_for_filename(std::time::SystemTime::now());
    let mut path = dir.join(format!("{stamp}-{label}.wav"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{stamp}-{label}-{n}.wav"));
        n += 1;
    }
    let mut wav = Vec::with_capacity(44 + bytes.len());
    wav.extend_from_slice(&wav_header(bytes.len() as u32));
    wav.extend_from_slice(&bytes);
    if let Err(e) = std::fs::write(&path, &wav) {
        log::warn!("Could not keep the recording at {}: {e}", path.display());
        return None;
    }
    let _ = std::fs::remove_file(&spool);
    Some(path)
}

/// Recordings, newest first.
pub fn list_recordings(data_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(recordings_dir(data_dir)) else {
        return Vec::new();
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "wav").unwrap_or(false))
        .map(|p| {
            let modified = std::fs::metadata(&p)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            (modified, p)
        })
        .collect();
    files.sort_by(|a, b| b.cmp(a));
    files.into_iter().map(|(_, p)| p).collect()
}

/// Retention of kept recordings: a week, at most 20 files, at most 200 MB
/// (a minute of 16 kHz 16-bit audio is 1.9 MB). Applied after every archive
/// and once at startup, so the folder never grows on its own.
pub const KEEP_FILES: usize = 20;
pub const KEEP_BYTES: u64 = 200 * 1024 * 1024;
pub const KEEP_AGE: std::time::Duration = std::time::Duration::from_secs(7 * 24 * 3600);

/// Drop recordings beyond `keep` files, `max_bytes` in total, or older than
/// `max_age`, oldest first.
pub fn prune_recordings(
    data_dir: &Path,
    keep: usize,
    max_bytes: u64,
    max_age: std::time::Duration,
) {
    let now = std::time::SystemTime::now();
    let mut total = 0u64;
    for (index, path) in list_recordings(data_dir).iter().enumerate() {
        let meta = std::fs::metadata(path).ok();
        let len = meta.as_ref().map(|m| m.len()).unwrap_or(0);
        let age = meta
            .and_then(|m| m.modified().ok())
            .and_then(|t| now.duration_since(t).ok())
            .unwrap_or_default();
        total += len;
        if index >= keep || total > max_bytes || age > max_age {
            match std::fs::remove_file(path) {
                Ok(()) => log::info!("Pruned old recording {}", path.display()),
                Err(e) => log::warn!("Could not prune {}: {e}", path.display()),
            }
        }
    }
}

/// [`prune_recordings`] with the built-in retention.
pub fn prune_recordings_default(data_dir: &Path) {
    prune_recordings(data_dir, KEEP_FILES, KEEP_BYTES, KEEP_AGE);
}

/// Read an archived recording (16 kHz mono 16-bit WAV) back into samples.
pub fn read_recording(path: &Path) -> Result<Vec<f32>, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("Could not read {}: {e}", path.display()))?;
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" {
        return Err(format!("{} is not a WAV file", path.display()));
    }
    Ok(bytes[44..]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / i16::MAX as f32)
        .collect())
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
    fn finished_spool_is_archived_as_wav_and_pruned() {
        use super::{archive_spool, list_recordings, prune_recordings, read_recording};
        let dir = test_dir("archive");
        let spool = spool_path(&dir);
        let bytes: Vec<u8> = (0..16000i16)
            .flat_map(|i| (i % 1000).to_le_bytes())
            .collect();
        std::fs::write(&spool, &bytes).unwrap();
        let archived = archive_spool(&dir, "pasted").expect("archived");
        assert!(!spool.exists(), "spool moved away");
        assert!(archived.starts_with(dir.join(super::RECORDINGS_DIR)));
        assert!(archived.extension().map(|e| e == "wav").unwrap_or(false));
        let wav = std::fs::read(&archived).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(wav.len(), 44 + bytes.len());
        assert_eq!(read_recording(&archived).expect("readable").len(), 16000);
        for i in 0..5 {
            std::fs::write(&spool, &bytes).unwrap();
            archive_spool(&dir, &format!("x{i}")).expect("archived");
        }
        assert_eq!(list_recordings(&dir).len(), 6);
        const WEEK: std::time::Duration = std::time::Duration::from_secs(7 * 24 * 3600);
        prune_recordings(&dir, 3, u64::MAX, WEEK);
        let left = list_recordings(&dir);
        assert_eq!(left.len(), 3, "{left:?}");
        let names: Vec<String> = left
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(
            names.iter().all(|n| n.contains("-x")),
            "newest kept: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.contains("pasted")),
            "oldest dropped: {names:?}"
        );
        prune_recordings(&dir, 10, 44 + bytes.len() as u64, WEEK);
        assert_eq!(
            list_recordings(&dir).len(),
            1,
            "byte cap keeps only the newest"
        );
        // Age: a file older than the limit goes even when the counts allow it.
        std::fs::write(&spool, &bytes).unwrap();
        let old = archive_spool(&dir, "old").expect("archived");
        let ten_days_ago =
            std::time::SystemTime::now() - std::time::Duration::from_secs(10 * 24 * 3600);
        std::fs::File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .set_modified(ten_days_ago)
            .unwrap();
        prune_recordings(&dir, 10, u64::MAX, WEEK);
        let left = list_recordings(&dir);
        assert_eq!(left.len(), 1, "{left:?}");
        assert!(
            !left[0].ends_with(old.file_name().unwrap()),
            "the aged file is gone"
        );
        assert!(
            archive_spool(&dir, "none").is_none(),
            "no spool, nothing archived"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

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
