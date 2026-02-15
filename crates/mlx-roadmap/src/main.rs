use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::ffi::OsStr;
use std::process::Command;
use std::time::Duration;
use thiserror::Error;

#[derive(Parser, Debug)]
#[command(name = "mlx-roadmap")]
#[command(about = "Repo automation helpers for oxidizedMLX", long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Create labels, milestones, and epic issues for the delivery plan (idempotent).
    Bootstrap {
        /// Target repo in OWNER/REPO format. If omitted, inferred from `git remote origin`.
        #[arg(long)]
        repo: Option<String>,

        /// Print what would change without making writes.
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Debug, Clone)]
struct Repo {
    owner: String,
    name: String,
}

impl Repo {
    fn slug(&self) -> String {
        format!("{}/{}", self.owner, self.name)
    }
}

#[derive(Error, Debug)]
enum Error {
    #[error("failed to run git: {0}")]
    Git(String),
    #[error("failed to run gh: {0}")]
    Gh(String),
    #[error("invalid repo slug (expected OWNER/REPO): {0}")]
    InvalidRepo(String),
    #[error("json parse error: {0}")]
    Json(String),
}

#[derive(Deserialize)]
struct Label {
    name: String,
}

#[derive(Deserialize)]
struct Milestone {
    number: u64,
    title: String,
}

#[derive(Deserialize)]
struct SearchResult {
    total_count: u64,
}

#[derive(Deserialize)]
struct Issue {
    number: u64,
}

fn main() -> Result<(), Error> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Bootstrap { repo, dry_run } => bootstrap(repo, dry_run),
    }
}

fn bootstrap(repo: Option<String>, dry_run: bool) -> Result<(), Error> {
    const MAX_ATTEMPTS: usize = 15;

    let repo = match repo {
        Some(s) => parse_repo_slug(&s)?,
        None => infer_repo_from_git_remote()?,
    };

    let (labels, milestones, epic_issues) = plan();

    let existing_labels: Vec<Label> = gh_json_with_retries(
        &["api", &format!("repos/{}/labels?per_page=100", repo.slug())],
        MAX_ATTEMPTS,
    )?;
    let existing_label_names: std::collections::BTreeSet<String> =
        existing_labels.into_iter().map(|l| l.name).collect();

    let mut created_labels = Vec::new();
    for l in labels {
        if existing_label_names.contains(l.name) {
            continue;
        }
        if dry_run {
            created_labels.push(l.name.to_string());
            continue;
        }
        gh_with_retries(
            &[
                "api",
                "-X",
                "POST",
                &format!("repos/{}/labels", repo.slug()),
                "-f",
                &format!("name={}", l.name),
                "-f",
                &format!("color={}", l.color),
                "-f",
                &format!("description={}", l.description),
            ],
            MAX_ATTEMPTS,
        )?;
        created_labels.push(l.name.to_string());
    }

    let existing_milestones: Vec<Milestone> = gh_json_with_retries(
        &[
            "api",
            &format!("repos/{}/milestones?state=all&per_page=100", repo.slug()),
        ],
        MAX_ATTEMPTS,
    )?;
    let mut milestones_by_title: std::collections::BTreeMap<String, Milestone> =
        existing_milestones
            .into_iter()
            .map(|m| (m.title.clone(), m))
            .collect();

    let mut created_milestones = Vec::new();
    for m in milestones {
        if milestones_by_title.contains_key(m.title) {
            continue;
        }
        if dry_run {
            created_milestones.push(m.title.to_string());
            continue;
        }
        let created: Milestone = gh_json_with_retries(
            &[
                "api",
                "-X",
                "POST",
                &format!("repos/{}/milestones", repo.slug()),
                "-f",
                &format!("title={}", m.title),
                "-f",
                &format!("description={}", m.description),
            ],
            MAX_ATTEMPTS,
        )?;
        created_milestones.push(created.title.clone());
        milestones_by_title.insert(created.title.clone(), created);
    }

    let mut created_issues = Vec::new();
    for epic in epic_issues {
        // Idempotence via search.
        let q = format!("repo:{} is:issue in:title \"{}\"", repo.slug(), epic.title);
        let search: SearchResult = gh_json_with_retries(
            &[
                "api",
                "search/issues",
                "-f",
                &format!("q={}", q),
                "-f",
                "per_page=5",
            ],
            MAX_ATTEMPTS,
        )?;
        if search.total_count > 0 {
            continue;
        }

        if dry_run {
            created_issues.push(epic.title.to_string());
            continue;
        }

        let ms_num = milestones_by_title.get(epic.title).map(|m| m.number);

        let mut args = vec![
            "api".to_string(),
            "-X".to_string(),
            "POST".to_string(),
            format!("repos/{}/issues", repo.slug()),
            "-f".to_string(),
            format!("title={}", epic.title),
            "-f".to_string(),
            format!("body={}", epic.body),
        ];
        if let Some(n) = ms_num {
            args.push("-f".to_string());
            args.push(format!("milestone={}", n));
        }

        let issue: Issue = gh_json_with_retries_owned(args, MAX_ATTEMPTS)?;

        // Apply labels.
        if !epic.labels.is_empty() {
            let mut lab_args = vec![
                "api".to_string(),
                "-X".to_string(),
                "POST".to_string(),
                format!("repos/{}/issues/{}/labels", repo.slug(), issue.number),
            ];
            for lbl in epic.labels {
                lab_args.push("-f".to_string());
                lab_args.push(format!("labels[]={}", lbl));
            }
            gh_with_retries_owned(lab_args, MAX_ATTEMPTS)?;
        }

        created_issues.push(format!("{} (#{})", epic.title, issue.number));
    }

    println!("repo: {}", repo.slug());
    if dry_run {
        println!("dry_run: true");
    }
    println!("created labels: {}", created_labels.len());
    for n in created_labels {
        println!("- {}", n);
    }
    println!("created milestones: {}", created_milestones.len());
    for n in created_milestones {
        println!("- {}", n);
    }
    println!("created epic issues: {}", created_issues.len());
    for n in created_issues {
        println!("- {}", n);
    }

    Ok(())
}

struct PlannedLabel {
    name: &'static str,
    color: &'static str,
    description: &'static str,
}

struct PlannedMilestone {
    title: &'static str,
    description: &'static str,
}

struct PlannedEpicIssue {
    title: &'static str,
    labels: &'static [&'static str],
    body: String,
}

fn plan() -> (
    Vec<PlannedLabel>,
    Vec<PlannedMilestone>,
    Vec<PlannedEpicIssue>,
) {
    let labels = vec![
        PlannedLabel {
            name: "epic",
            color: "0e8a16",
            description: "Umbrella tracking issue (milestone-level).",
        },
        PlannedLabel {
            name: "dx",
            color: "1d76db",
            description: "Developer experience (CI, docs, tooling).",
        },
        PlannedLabel {
            name: "backend",
            color: "5319e7",
            description: "Backend work (CPU/Metal/FFI).",
        },
        PlannedLabel {
            name: "conformance",
            color: "c2e0c6",
            description: "Rust vs Python MLX conformance harness.",
        },
        PlannedLabel {
            name: "autograd",
            color: "fbca04",
            description: "Reverse-mode AD / gradients.",
        },
        PlannedLabel {
            name: "nn",
            color: "b60205",
            description: "Neural network modules / parameters.",
        },
        PlannedLabel {
            name: "optim",
            color: "0052cc",
            description: "Optimizers and schedulers.",
        },
        PlannedLabel {
            name: "io",
            color: "006b75",
            description: "Serialization / safetensors / loading.",
        },
    ];

    let milestones = vec![
        PlannedMilestone {
            title: "M0: Repo Baseline",
            description: "Docs + CI/DX baseline.",
        },
        PlannedMilestone {
            title: "M1: Backend Trait + Dispatch",
            description: "Define the narrow-waist backend abstraction and route core ops through it.",
        },
        PlannedMilestone {
            title: "M2: Core Tensor API Stabilization",
            description: "Stabilize public tensor API, error types, dtype/shape rules.",
        },
        PlannedMilestone {
            title: "M3: Ops Coverage",
            description: "Expand pure ops layer (elementwise, reductions, matmul utilities).",
        },
        PlannedMilestone {
            title: "M4: CPU Backend",
            description: "Correctness-first reference backend implementation.",
        },
        PlannedMilestone {
            title: "M5: Autograd",
            description: "Reverse-mode autodiff engine and VJP registry.",
        },
        PlannedMilestone {
            title: "M6: Metal Backend Scaffolding",
            description: "Metal runtime scaffolding and first end-to-end op.",
        },
        PlannedMilestone {
            title: "M7: Conformance Harness",
            description: "Rust vs Python MLX comparisons + reports.",
        },
        PlannedMilestone {
            title: "M8: FFI Backend",
            description: "Upstream MLX via versioned C ABI shim and backend integration.",
        },
        PlannedMilestone {
            title: "M9: NN / Optim / IO",
            description: "NN modules, optimizers, and safetensors/mmap IO.",
        },
    ];

    let notes = [
        "Repo docs:",
        "- `docs/DELIVERY_PLAN.md`",
        "- `docs/REPO_CHECKLIST.md`",
        "",
        "Definition of done:",
        "- Tests + docs updated.",
        "- Conformance story satisfied where applicable.",
    ];

    let epic_issues = [
        (
            "M0: Repo Baseline",
            &["epic", "dx"][..],
            &[
                "- [ ] Add top-level `README.md` and `docs/*` (delivery plan + checklist).",
                "- [ ] Ensure `just ci` passes on a fresh clone without `MLX_SRC`.",
                "- [ ] Add `CONTRIBUTING.md` with dev workflows (optional).",
            ][..],
        ),
        (
            "M1: Backend Trait + Dispatch",
            &["epic", "backend"][..],
            &[
                "- [ ] Define backend trait contract in `crates/mlx-core` (or a dedicated crate) including dtype/shape/device semantics.",
                "- [ ] Route a minimal op set through backend dispatch: add/mul/matmul/sum.",
                "- [ ] Add backend selection at tensor creation time.",
            ][..],
        ),
        (
            "M2: Core Tensor API Stabilization",
            &["epic"][..],
            &[
                "- [ ] Consolidate error types (`thiserror`) and enforce invariants (rank checks, dtype checks).",
                "- [ ] Define dtype policy (f16/bf16/f32/f64/i32/i64/bool) and conversions.",
                "- [ ] Add property tests for broadcasting + shape rules (proptest).",
            ][..],
        ),
        (
            "M3: Ops Coverage",
            &["epic"][..],
            &[
                "- [ ] Elementwise: add/sub/mul/div/neg/exp/log/tanh/relu-ish.",
                "- [ ] Reductions: sum/mean/max/min with axis handling.",
                "- [ ] Matmul and basic linear algebra utilities.",
            ][..],
        ),
        (
            "M4: CPU Backend",
            &["epic", "backend"][..],
            &[
                "Sources: `crates/mlx-cpu/src/lib.rs` has `TODO(milestone-4)`.",
                "",
                "- [ ] Implement backend trait for CPU: tensor storage, strides, basic kernels.",
                "- [ ] Implement broadcast + elementwise kernels.",
                "- [ ] Implement matmul kernel (naive first), plus test vectors.",
                "- [ ] Add deterministic RNG policy for tests (seeded).",
            ][..],
        ),
        (
            "M5: Autograd",
            &["epic", "autograd"][..],
            &[
                "Sources: `crates/mlx-autograd/src/lib.rs` has `TODO(milestone-5)`.",
                "",
                "- [ ] Define tape/graph representation and gradient accumulation policy.",
                "- [ ] Add VJP registry for core ops (matmul, sum, add, mul).",
                "- [ ] Gradient correctness tests (finite differences where feasible).",
            ][..],
        ),
        (
            "M6: Metal Backend Scaffolding",
            &["epic", "backend"][..],
            &[
                "Sources: `crates/mlx-metal/src/lib.rs` has `TODO(milestone-6)`.",
                "",
                "- [ ] Add Metal runtime scaffolding: device/queue/buffer abstractions.",
                "- [ ] Implement one op end-to-end on Metal with conformance vs CPU.",
                "- [ ] Add a feature flag `metal` and guard macOS-only code.",
            ][..],
        ),
        (
            "M7: Conformance Harness",
            &["epic", "conformance"][..],
            &[
                "- [ ] Define a conformance test spec (ops, shapes, dtypes, tolerances).",
                "- [ ] Implement harness in `crates/mlx-conformance` that runs Python MLX and compares.",
                "- [ ] Add a report format that includes repro inputs and expected/got deltas.",
            ][..],
        ),
        (
            "M8: FFI Backend",
            &["epic", "backend"][..],
            &[
                "Sources: `crates/mlx-sys/build.rs` uses `MLX_SRC` and builds a static `mlxrs_capi`.",
                "",
                "- [ ] Document and version the C ABI surface (functions + types) in the shim.",
                "- [ ] Add an `ffi` feature and an FFI backend implementation behind the backend trait.",
                "- [ ] Add a CI job that runs FFI builds behind an opt-in workflow (or required secrets/checkout).",
            ][..],
        ),
        (
            "M9: NN / Optim / IO",
            &["epic", "nn", "optim", "io"][..],
            &[
                "Sources:",
                "- `crates/mlx-nn/src/lib.rs` has `TODO(milestone-9)`.",
                "- `crates/mlx-optim/src/lib.rs` has `TODO(milestone-9)`.",
                "- `crates/mlx-io/src/lib.rs` has `TODO(milestone-9)`.",
                "",
                "- [ ] Implement module/parameter system and state dict conventions (`mlx-nn`).",
                "- [ ] Implement SGD + AdamW + schedulers (`mlx-optim`).",
                "- [ ] Implement safetensors + mmap loader (`mlx-io`) with validation and tests.",
            ][..],
        ),
    ];

    let epics = epic_issues
        .into_iter()
        .map(|(title, labels, lines)| PlannedEpicIssue {
            title,
            labels,
            body: {
                let mut out = String::new();
                out.push_str(&format!("## {title}\n\nTracking checklist:\n\n"));
                for l in lines {
                    out.push_str(l);
                    out.push('\n');
                }
                out.push('\n');
                for n in notes {
                    out.push_str(n);
                    out.push('\n');
                }
                out
            },
        })
        .collect::<Vec<_>>();

    (labels, milestones, epics)
}

fn parse_repo_slug(s: &str) -> Result<Repo, Error> {
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        return Err(Error::InvalidRepo(s.to_string()));
    }
    Ok(Repo {
        owner: parts[0].to_string(),
        name: parts[1].to_string(),
    })
}

fn infer_repo_from_git_remote() -> Result<Repo, Error> {
    let out = run_capture("git", ["config", "--get", "remote.origin.url"]).map_err(Error::Git)?;
    let url = String::from_utf8_lossy(&out).trim().to_string();
    if url.is_empty() {
        return Err(Error::Git("remote.origin.url is empty".to_string()));
    }
    // Supports:
    // - https://github.com/OWNER/REPO(.git)
    // - git@github.com:OWNER/REPO(.git)
    let path = if let Some(rest) = url.strip_prefix("https://github.com/") {
        rest.to_string()
    } else if let Some(rest) = url.strip_prefix("git@github.com:") {
        rest.to_string()
    } else {
        return Err(Error::Git(format!("unsupported origin url: {url}")));
    };
    let path = path.strip_suffix(".git").unwrap_or(&path);
    parse_repo_slug(path)
}

fn is_transient_gh_error(stderr: &str) -> bool {
    stderr.contains("error connecting to api.github.com")
        || stderr.contains("check your internet connection")
        || stderr.contains("githubstatus.com")
}

fn gh_with_retries(args: &[&str], max_attempts: usize) -> Result<Vec<u8>, Error> {
    gh_with_retries_owned(args.iter().map(|s| s.to_string()).collect(), max_attempts)
}

fn gh_with_retries_owned(args: Vec<String>, max_attempts: usize) -> Result<Vec<u8>, Error> {
    for attempt in 1..=max_attempts {
        let mut cmd = Command::new("gh");
        cmd.args(&args);
        let output = cmd.output().map_err(|e| Error::Gh(e.to_string()))?;
        if output.status.success() {
            return Ok(output.stdout);
        }
        // gh sometimes prints connection errors to stdout, so consider both.
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let combined = format!("{stdout}{stderr}");
        if attempt < max_attempts && is_transient_gh_error(&combined) {
            // Exponential backoff with deterministic jitter.
            // (No RNG dependency; this is just to avoid hammering.)
            let base_ms = 200u64.saturating_mul(1u64 << attempt.min(4));
            let jitter_ms = (attempt as u64 * 137) % 200;
            let sleep_ms = (base_ms + jitter_ms).min(5_000);
            eprintln!(
                "gh retry {attempt}/{max_attempts} after transient error (sleep {sleep_ms}ms): {}",
                combined.lines().next().unwrap_or("<no output>")
            );
            std::thread::sleep(Duration::from_millis(sleep_ms));
            continue;
        }
        if is_transient_gh_error(&combined) {
            return Err(Error::Gh(format!(
                "{combined}\n(transient GitHub API connectivity error; retry again in a few seconds)"
            )));
        }
        return Err(Error::Gh(combined));
    }
    Err(Error::Gh("exhausted retries".to_string()))
}

fn gh_json_with_retries<T: for<'de> Deserialize<'de>>(
    args: &[&str],
    max_attempts: usize,
) -> Result<T, Error> {
    let stdout = gh_with_retries(args, max_attempts)?;
    serde_json::from_slice(&stdout).map_err(|e| Error::Json(e.to_string()))
}

fn gh_json_with_retries_owned<T: for<'de> Deserialize<'de>>(
    args: Vec<String>,
    max_attempts: usize,
) -> Result<T, Error> {
    let stdout = gh_with_retries_owned(args, max_attempts)?;
    serde_json::from_slice(&stdout).map_err(|e| Error::Json(e.to_string()))
}

fn run_capture<I, S>(bin: &str, args: I) -> Result<Vec<u8>, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let out = Command::new(bin)
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    if out.status.success() {
        return Ok(out.stdout);
    }
    Err(String::from_utf8_lossy(&out.stderr).to_string())
}
