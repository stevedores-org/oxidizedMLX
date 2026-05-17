use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDiscovery {
    pub name: String,
    pub path: PathBuf,
    pub framework: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryReport {
    pub generated_at: String,
    pub service_count: usize,
    pub services: Vec<ServiceDiscovery>,
}

pub fn discover_services(root: &Path) -> Result<Vec<ServiceDiscovery>> {
    let mut services = Vec::new();
    for search_dir in ["apps", "apps-agents", "crates"] {
        let base = root.join(search_dir);
        if !base.exists() {
            continue;
        }

        for entry in WalkDir::new(&base)
            .max_depth(3)
            .into_iter()
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if path.file_name().is_some_and(|name| name == "Cargo.toml") {
                let content = fs::read_to_string(path)?;
                if let Some(framework) = detect_rust_framework(&content)
                    && let Some(parent) = path.parent()
                {
                    let name = cargo_package_name(&content).unwrap_or_else(|| {
                        parent
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("unknown-service")
                            .to_string()
                    });
                    services.push(ServiceDiscovery {
                        name,
                        path: parent.to_path_buf(),
                        framework,
                    });
                }
            }
        }
    }
    services.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(services)
}

pub fn discovery_report(services: Vec<ServiceDiscovery>) -> DiscoveryReport {
    DiscoveryReport {
        generated_at: chrono::Utc::now().to_rfc3339(),
        service_count: services.len(),
        services,
    }
}

pub fn generate_openapi_template(service: &ServiceDiscovery, repo_root: &Path) -> Value {
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": service.name,
            "version": "0.1.0",
            "description": format!("API specification template for {} ({})", service.name, service.framework),
        },
        "paths": {},
        "components": {
            "schemas": {}
        },
        "servers": [
            {
                "url": "http://localhost:8080",
                "description": "Local development server"
            }
        ],
        "x-generated": {
            "tool": "oxidizedMLX openapi-extractor",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "service_path": relative_path(repo_root, &service.path),
            "framework": service.framework,
            "note": "Generated template; populate route details from service code or an OpenAPI derive integration."
        }
    })
}

pub fn write_outputs(root: &Path, output_dir: &Path, report: &DiscoveryReport) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    let report_path = output_dir.join("openapi-discovery.json");
    fs::write(&report_path, serde_json::to_string_pretty(report)?)?;

    for service in &report.services {
        let spec = generate_openapi_template(service, root);
        let spec_path = output_dir.join(format!("{}.openapi.json", service.name));
        fs::write(spec_path, serde_json::to_string_pretty(&spec)?)?;
    }
    Ok(())
}

fn detect_rust_framework(cargo_toml: &str) -> Option<String> {
    for framework in ["axum", "actix-web", "rocket"] {
        if cargo_toml.contains(framework) {
            return Some(framework.to_string());
        }
    }
    None
}

fn cargo_package_name(cargo_toml: &str) -> Option<String> {
    let mut in_package = false;
    for line in cargo_toml.lines() {
        let trimmed = line.trim();
        if trimmed == "[package]" {
            in_package = true;
            continue;
        }
        if in_package && trimmed.starts_with('[') {
            return None;
        }
        if in_package && trimmed.starts_with("name") {
            return trimmed
                .split_once('=')
                .map(|(_, value)| value.trim().trim_matches('"').to_string());
        }
    }
    None
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_axum_services() {
        let cargo = r#"
            [package]
            name = "svc"
            [dependencies]
            axum = "0.8"
        "#;
        assert_eq!(detect_rust_framework(cargo).as_deref(), Some("axum"));
        assert_eq!(cargo_package_name(cargo).as_deref(), Some("svc"));
    }
}
