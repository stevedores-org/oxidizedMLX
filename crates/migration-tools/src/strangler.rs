use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StranglerConfig {
    pub legacy_roots: Vec<String>,
    #[serde(default)]
    pub descriptions: HashMap<String, String>,
}

impl StranglerConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PathClassification {
    pub path: String,
    pub legacy_root: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SizeReport {
    pub legacy_root: String,
    pub bytes: u64,
    pub human_size: String,
    pub status: String,
}

pub struct Strangler {
    config: StranglerConfig,
    repo_root: PathBuf,
}

impl Strangler {
    pub fn new(config: StranglerConfig, repo_root: PathBuf) -> Self {
        Self { config, repo_root }
    }

    pub fn repo_root() -> Result<PathBuf> {
        let output = Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .output()?;
        if output.status.success() {
            Ok(PathBuf::from(String::from_utf8(output.stdout)?.trim()))
        } else {
            Ok(std::env::current_dir()?)
        }
    }

    pub fn diff_paths(&self, base: &str) -> Result<Vec<String>> {
        for separator in ["...", ".."] {
            let spec = format!("{base}{separator}HEAD");
            let output = Command::new("git")
                .args(["diff", "--name-only", &spec])
                .current_dir(&self.repo_root)
                .output()?;
            if output.status.success() {
                return parse_paths(output.stdout);
            }
        }
        Ok(Vec::new())
    }

    pub fn tracked_paths(&self) -> Result<Vec<String>> {
        let output = Command::new("git")
            .args(["ls-files"])
            .current_dir(&self.repo_root)
            .output()?;
        if output.status.success() {
            parse_paths(output.stdout)
        } else {
            Ok(Vec::new())
        }
    }

    pub fn classify(&self, paths: &[String]) -> Vec<PathClassification> {
        let mut hits = Vec::new();
        for path in paths {
            let normalized = normalize(path);
            for root in &self.config.legacy_roots {
                let normalized_root = normalize(root);
                if normalized == normalized_root
                    || normalized.starts_with(&format!("{normalized_root}/"))
                {
                    hits.push(PathClassification {
                        path: path.clone(),
                        legacy_root: root.clone(),
                    });
                    break;
                }
            }
        }
        hits
    }

    pub fn size_report(&self) -> Vec<SizeReport> {
        self.config
            .legacy_roots
            .iter()
            .map(|root| {
                let target = self.repo_root.join(root);
                if !target.exists() {
                    return SizeReport {
                        legacy_root: root.clone(),
                        bytes: 0,
                        human_size: "missing".to_string(),
                        status: "missing".to_string(),
                    };
                }
                match directory_size(&target) {
                    Ok(bytes) => SizeReport {
                        legacy_root: root.clone(),
                        bytes,
                        human_size: human_size(bytes),
                        status: "ok".to_string(),
                    },
                    Err(error) => SizeReport {
                        legacy_root: root.clone(),
                        bytes: 0,
                        human_size: "error".to_string(),
                        status: error.to_string(),
                    },
                }
            })
            .collect()
    }
}

fn parse_paths(stdout: Vec<u8>) -> Result<Vec<String>> {
    Ok(String::from_utf8(stdout)?
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect())
}

fn normalize(path: &str) -> String {
    Path::new(path)
        .to_string_lossy()
        .trim_start_matches("./")
        .to_string()
}

fn directory_size(path: &Path) -> std::io::Result<u64> {
    let mut total = 0;
    for entry in walkdir::WalkDir::new(path).into_iter().flatten() {
        if entry.path().is_file() {
            total += entry.metadata()?.len();
        }
    }
    Ok(total)
}

pub fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = bytes as f64;
    for unit in UNITS {
        if size < 1024.0 {
            return format!("{size:.1}{unit}");
        }
        size /= 1024.0;
    }
    format!("{size:.1}TB")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_legacy_paths() {
        let config = StranglerConfig {
            legacy_roots: vec!["tools/py_ref_runner".to_string()],
            descriptions: HashMap::new(),
        };
        let strangler = Strangler::new(config, PathBuf::from("."));
        let hits = strangler.classify(&[
            "tools/py_ref_runner/run.py".to_string(),
            "crates/mlx-core/src/lib.rs".to_string(),
        ]);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].legacy_root, "tools/py_ref_runner");
    }
}
