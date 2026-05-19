use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use toml::Value as TomlValue;
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
                let manifest = content.parse::<TomlValue>()?;
                if let Some(framework) = detect_rust_framework(&manifest)
                    && let Some(parent) = path.parent()
                {
                    let name = cargo_package_name(&manifest).unwrap_or_else(|| {
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

fn detect_rust_framework(cargo_toml: &TomlValue) -> Option<String> {
    let dependencies = cargo_toml.get("dependencies")?.as_table()?;
    // Deterministic priority when a service exposes more than one HTTP framework.
    for framework in ["axum", "actix-web", "rocket"] {
        if dependencies.contains_key(framework) {
            return Some(framework.to_string());
        }
    }
    None
}

fn cargo_package_name(cargo_toml: &TomlValue) -> Option<String> {
    cargo_toml
        .get("package")?
        .get("name")?
        .as_str()
        .map(ToOwned::to_owned)
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
        let manifest = cargo.parse::<TomlValue>().unwrap();

        assert_eq!(detect_rust_framework(&manifest).as_deref(), Some("axum"));
        assert_eq!(cargo_package_name(&manifest).as_deref(), Some("svc"));
    }

    #[test]
    fn ignores_comments_and_dev_dependencies() {
        let cargo = r#"
            [package]
            name = "not-a-service"

            # axum = "0.8"
            [dev-dependencies]
            rocket = "0.5"
        "#;
        let manifest = cargo.parse::<TomlValue>().unwrap();

        assert_eq!(detect_rust_framework(&manifest), None);
    }

    #[test]
    fn discovers_services_from_fixture_tree() {
        let temp = tempfile::tempdir().unwrap();
        let service_dir = temp.path().join("crates/api");
        fs::create_dir_all(&service_dir).unwrap();
        fs::write(
            service_dir.join("Cargo.toml"),
            r#"
            [package]
            name = "api"

            [dependencies]
            axum = "0.8"
            "#,
        )
        .unwrap();

        let services = discover_services(temp.path()).unwrap();

        assert_eq!(services.len(), 1);
        assert_eq!(services[0].name, "api");
        assert_eq!(services[0].framework, "axum");
    }

    #[test]
    fn write_outputs_creates_report_and_specs() {
        let temp = tempfile::tempdir().unwrap();
        let output_dir = temp.path().join("out");
        let service_dir = temp.path().join("crates/api");
        fs::create_dir_all(&service_dir).unwrap();
        let report = discovery_report(vec![ServiceDiscovery {
            name: "api".to_string(),
            path: service_dir,
            framework: "axum".to_string(),
        }]);

        write_outputs(temp.path(), &output_dir, &report).unwrap();

        assert!(output_dir.join("openapi-discovery.json").is_file());
        assert!(output_dir.join("api.openapi.json").is_file());
    }
}
