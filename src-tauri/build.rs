use std::process::Command;

/// Bake the git state into the binary so wispr.log and the About line say
/// which build is running (weeks of uncommitted changes were otherwise
/// indistinguishable from the last tag).
fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn main() {
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/index");

    let hash = git(&["rev-parse", "--short=9", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
        .map(|s| if s.is_empty() { "" } else { "-dirty" })
        .unwrap_or("");
    let built = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    println!("cargo:rustc-env=WISPR_GIT_HASH={hash}");
    println!("cargo:rustc-env=WISPR_GIT_DIRTY={dirty}");
    println!("cargo:rustc-env=WISPR_BUILD_TIME={built}");

    tauri_build::build()
}
