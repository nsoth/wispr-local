use cpal::traits::{DeviceTrait, HostTrait};

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
    devices
}

pub fn get_default_input_device() -> Option<(cpal::Device, cpal::SupportedStreamConfig)> {
    let host = cpal::default_host();
    let device = host.default_input_device()?;
    let config = device.default_input_config().ok()?;
    Some((device, config))
}
