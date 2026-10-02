//! Input device enumeration and selection.
//!
//! Windows renumbers a USB/wireless microphone after a re-plug: the device the
//! user picked as "Microphone (Wireless Mic Rx)" comes back as
//! "Microphone (2- Wireless Mic Rx)". cpal 0.15 exposes only display names,
//! so matching tolerates that prefix instead of falling back to the system
//! default (and announcing it) on almost every recording.

use cpal::traits::{DeviceTrait, HostTrait};

#[derive(Debug, Clone, serde::Serialize)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
    /// This device is the current Windows default input.
    pub is_default: bool,
}

/// Strip the "N- " that Windows prepends inside the parentheses of a
/// re-enumerated device ("Microphone (2- HyperX SoloCast)").
pub fn normalize_device_name(name: &str) -> String {
    let Some(open) = name.find('(') else {
        return name.to_string();
    };
    let inner = &name[open + 1..];
    let digits = inner.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 && inner[digits..].starts_with("- ") {
        format!("{}{}", &name[..open + 1], &inner[digits + 2..])
    } else {
        name.to_string()
    }
}

/// Index of the device matching `preferred`: exact name first, then equal
/// normalized names. `None` when nothing matches.
pub fn pick_device(preferred: &str, names: &[String]) -> Option<usize> {
    if let Some(index) = names.iter().position(|n| n == preferred) {
        return Some(index);
    }
    let wanted = normalize_device_name(preferred);
    names
        .iter()
        .position(|n| normalize_device_name(n) == wanted)
}

/// Whether a fallback to another microphone should be announced to the user:
/// only the first time a given device is used as the fallback.
pub fn should_announce_fallback(last_announced: Option<&str>, device: &str) -> bool {
    last_announced != Some(device)
}

pub fn list_input_devices() -> Vec<AudioDeviceInfo> {
    let host = cpal::default_host();
    let default_name = host
        .default_input_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();
    let mut devices = Vec::new();
    if let Ok(input_devices) = host.input_devices() {
        for device in input_devices {
            if let Ok(config) = device.default_input_config() {
                let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
                devices.push(AudioDeviceInfo {
                    is_default: name == default_name,
                    name,
                    sample_rate: config.sample_rate().0,
                    channels: config.channels(),
                });
            }
        }
    }
    devices.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    devices.dedup_by(|a, b| a.name == b.name);
    devices
}

/// How the device for a recording was chosen.
pub struct SelectedDevice {
    pub device: cpal::Device,
    pub config: cpal::SupportedStreamConfig,
    pub name: String,
    /// The preferred device was not found (or could not be opened) and the
    /// system default is used instead.
    pub used_fallback: bool,
    /// Why the preferred device was not used, for the log line.
    pub fallback_reason: Option<String>,
}

/// Resolve the user's preferred microphone, tolerating Windows' renumbering,
/// and fall back to the system default when it is absent.
pub fn select_input_device(preferred_name: Option<&str>) -> Result<SelectedDevice, String> {
    let host = cpal::default_host();
    let preferred = preferred_name
        .map(str::trim)
        .filter(|name| !name.is_empty());
    let mut fallback_reason = None;

    if let Some(preferred) = preferred {
        let input_devices: Vec<cpal::Device> = host
            .input_devices()
            .map_err(|e| format!("Failed to list input devices: {e}"))?
            .collect();
        let names: Vec<String> = input_devices
            .iter()
            .map(|d| d.name().unwrap_or_default())
            .collect();
        match pick_device(preferred, &names) {
            Some(index) => {
                let device = &input_devices[index];
                match device.default_input_config() {
                    Ok(config) => {
                        return Ok(SelectedDevice {
                            device: device.clone(),
                            config,
                            name: names[index].clone(),
                            used_fallback: false,
                            fallback_reason: None,
                        })
                    }
                    Err(e) => {
                        fallback_reason = Some(format!("'{preferred}' could not be opened: {e}"));
                    }
                }
            }
            None => {
                fallback_reason = Some(format!("'{preferred}' is not connected"));
            }
        }
    }

    let device = host.default_input_device().ok_or("No input device found")?;
    let name = device
        .name()
        .unwrap_or_else(|_| "System default".to_string());
    let config = device
        .default_input_config()
        .map_err(|e| format!("Failed to get default input config: {e}"))?;
    Ok(SelectedDevice {
        device,
        config,
        name,
        used_fallback: fallback_reason.is_some(),
        fallback_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::{normalize_device_name, pick_device, should_announce_fallback};

    #[test]
    fn strips_the_windows_renumbering_prefix() {
        assert_eq!(
            normalize_device_name("Microphone (2- Wireless Mic Rx)"),
            "Microphone (Wireless Mic Rx)"
        );
        assert_eq!(
            normalize_device_name("Microphone Array (12- Intel® Smart Sound)"),
            "Microphone Array (Intel® Smart Sound)"
        );
        assert_eq!(
            normalize_device_name("Microphone (NVIDIA Broadcast)"),
            "Microphone (NVIDIA Broadcast)"
        );
        // Only a digit run followed by "- " right after "(" is a prefix.
        assert_eq!(normalize_device_name("Mic (A-1 Pro)"), "Mic (A-1 Pro)");
    }

    #[test]
    fn exact_match_wins_over_normalized_match() {
        let names = vec![
            "Microphone (2- HyperX SoloCast)".to_string(),
            "Microphone (HyperX SoloCast)".to_string(),
        ];
        assert_eq!(pick_device("Microphone (HyperX SoloCast)", &names), Some(1));
    }

    #[test]
    fn renumbered_device_still_matches() {
        let names = vec![
            "Microphone (NVIDIA Broadcast)".to_string(),
            "Microphone (2- Wireless Mic Rx)".to_string(),
        ];
        assert_eq!(pick_device("Microphone (Wireless Mic Rx)", &names), Some(1));
        assert_eq!(
            pick_device("Microphone (3- Wireless Mic Rx)", &names),
            Some(1)
        );
        assert_eq!(pick_device("Microphone (Blue Yeti)", &names), None);
    }

    #[test]
    fn fallback_is_announced_once_per_device() {
        assert!(should_announce_fallback(None, "Microphone (HyperX)"));
        assert!(!should_announce_fallback(
            Some("Microphone (HyperX)"),
            "Microphone (HyperX)"
        ));
        assert!(should_announce_fallback(
            Some("Microphone (HyperX)"),
            "Microphone (NVIDIA Broadcast)"
        ));
    }
}
