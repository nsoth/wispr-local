use cpal::traits::{DeviceTrait, HostTrait};

#[derive(Debug, Clone, serde::Serialize)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
}

pub fn list_input_devices() -> Vec<AudioDeviceInfo> {
    let host = cpal::default_host();
    let mut devices = Vec::new();
    if let Ok(input_devices) = host.input_devices() {
        for device in input_devices {
            if let Ok(config) = device.default_input_config() {
                devices.push(AudioDeviceInfo {
                    name: device.name().unwrap_or_else(|_| "Unknown".to_string()),
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

pub fn select_input_device(
    preferred_name: Option<&str>,
) -> Result<(cpal::Device, cpal::SupportedStreamConfig, String, bool), String> {
    let host = cpal::default_host();
    if let Some(preferred) = preferred_name.filter(|name| !name.trim().is_empty()) {
        let input_devices = host
            .input_devices()
            .map_err(|e| format!("Failed to list input devices: {e}"))?;
        for device in input_devices {
            if device.name().ok().as_deref() == Some(preferred) {
                let config = device
                    .default_input_config()
                    .map_err(|e| format!("Selected microphone is unavailable: {e}"))?;
                return Ok((device, config, preferred.to_string(), false));
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
    Ok((
        device,
        config,
        name,
        preferred_name.is_some_and(|name| !name.trim().is_empty()),
    ))
}
