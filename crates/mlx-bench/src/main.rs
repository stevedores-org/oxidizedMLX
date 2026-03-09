mod backend;
mod error;
mod report;
mod runner;
mod task;

use backend::{AnthropicApiBackend, BackendOpts, LocalMlxBackend, StdinDebugBackend, LlmBackend};
use clap::{Parser, Subcommand};
use error::Result;
use report::{ReportFormat, Reporter};
use runner::{EvalResult, RunConfig, TaskRunner};
use std::fs;
use std::path::PathBuf;
use task::TaskSet;

#[derive(Parser)]
#[command(name = "mlx-bench")]
#[command(about = "SWE-bench style LLM evaluation for oxidizedMLX")]
struct Args {
    /// Root workspace directory
    #[arg(long, default_value = ".")]
    workspace: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List all available evaluation tasks
    ListTasks {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show details of a specific task
    ShowTask {
        /// Task ID
        task_id: String,

        /// Show context file contents
        #[arg(long)]
        show_context: bool,
    },

    /// Validate all task JSON files
    ValidateTasks,

    /// Run self-test on golden patches
    SelfTest {
        /// Only test matching task IDs (glob pattern)
        #[arg(long)]
        filter: Option<String>,
    },

    /// Run benchmark against tasks
    Run {
        /// Backend to use: local, anthropic, debug
        #[arg(long, default_value = "anthropic")]
        backend: String,

        /// Model ID for LLM (e.g., claude-3-5-sonnet-20241022)
        #[arg(long)]
        model: Option<String>,

        /// Number of attempts per task
        #[arg(long, short = 'k', default_value_t = 1)]
        attempts: usize,

        /// Task ID glob filter
        #[arg(long)]
        filter: Option<String>,

        /// Dry run: print config but don't execute
        #[arg(long)]
        dry_run: bool,
    },

    /// Generate evaluation report
    Report {
        /// Output format: table, json, markdown
        #[arg(long, default_value = "table")]
        format: String,

        /// Use latest results
        #[arg(long)]
        latest: bool,

        /// Results file path
        #[arg(long)]
        file: Option<PathBuf>,
    },
}

fn main() {
    let args = Args::parse();

    if let Err(e) = run(args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    let tasks = TaskSet::load_from_dir(&args.workspace)?;

    match args.cmd {
        Cmd::ListTasks { json } => cmd_list_tasks(&tasks, json),
        Cmd::ShowTask { task_id, show_context } => {
            cmd_show_task(&tasks, &task_id, show_context, &args.workspace)
        }
        Cmd::ValidateTasks => cmd_validate_tasks(&tasks),
        Cmd::SelfTest { filter } => cmd_self_test(&tasks, filter.as_deref(), &args.workspace),
        Cmd::Run {
            backend,
            model,
            attempts,
            filter,
            dry_run,
        } => cmd_run(
            &tasks,
            &backend,
            model.as_deref(),
            attempts,
            filter.as_deref(),
            dry_run,
            &args.workspace,
        ),
        Cmd::Report { format, latest, file } => {
            cmd_report(&format, latest, file.as_deref())
        }
    }
}

fn cmd_list_tasks(tasks: &TaskSet, json: bool) -> Result<()> {
    if json {
        let json = serde_json::to_string_pretty(tasks.all())?;
        println!("{}", json);
    } else {
        println!("Available Tasks:\n");
        for task in tasks.all() {
            println!("  {} - {}", task.id, task.title);
            if let Some(issue) = task.issue {
                println!("    Issue: #{}", issue);
            }
        }
    }
    Ok(())
}

fn cmd_show_task(
    tasks: &TaskSet,
    task_id: &str,
    show_context: bool,
    workspace_root: &std::path::Path,
) -> Result<()> {
    let task = tasks
        .by_id(task_id)
        .ok_or_else(|| error::BenchError::TaskNotFound(task_id.to_string()))?;

    println!("Task: {}", task.id);
    println!("Title: {}", task.title);
    if let Some(issue) = task.issue {
        println!("Issue: #{}", issue);
    }
    if let Some(commit) = &task.fix_commit {
        println!("Fix Commit: {}", commit);
    }
    println!("\nDescription:\n{}\n", task.description);
    println!("Context Files:");
    for ctx in &task.context_files {
        println!("  - {}", ctx.path);
        if let Some((start, end)) = ctx.lines {
            println!("    Lines: {}-{}", start, end);
        }
        if let Some(ann) = &ctx.annotation {
            println!("    Annotation: {}", ann);
        }
    }
    println!("\nTest Filters: {:?}", task.test_filters);
    println!("Test Crates: {:?}", task.test_crates);
    println!("Timeout: {}s", task.timeout_secs);

    if show_context {
        println!("\n=== CONTEXT ===\n");
        for ctx in &task.context_files {
            match ctx.read_content(workspace_root) {
                Ok(content) => {
                    println!("--- {} ---", ctx.path);
                    println!("{}\n", content);
                }
                Err(e) => {
                    eprintln!("Failed to read {}: {}", ctx.path, e);
                }
            }
        }
    }

    Ok(())
}

fn cmd_validate_tasks(tasks: &TaskSet) -> Result<()> {
    tasks.validate()?;
    println!("✓ All {} tasks valid", tasks.all().len());
    Ok(())
}

fn cmd_self_test(
    tasks: &TaskSet,
    filter: Option<&str>,
    workspace_root: &std::path::Path,
) -> Result<()> {
    let to_test: Vec<_> = if let Some(f) = filter {
        tasks.by_glob(f)
    } else {
        tasks.all().iter().collect()
    };

    let config = RunConfig::new(workspace_root);
    let runner = TaskRunner::new(config);

    println!("Running self-tests on {} tasks...\n", to_test.len());

    for task in &to_test {
        if let Some(golden_patch) = &task.golden_patch {
            println!("Testing {} - {}", task.id, task.title);

            // Set patch in environment
            unsafe {
                std::env::set_var("BENCH_PATCH_FILE", "/tmp/golden_patch.diff");
            }
            fs::write("/tmp/golden_patch.diff", golden_patch)?;

            let opts = BackendOpts {
                model_id: "".to_string(),
                max_tokens: 0,
                temperature: 0.0,
                timeout_secs: task.timeout_secs,
            };

            let debug_backend = StdinDebugBackend;
            match runner.run_task(task, &debug_backend, &opts) {
                Ok(outcomes) => {
                    if let Some(outcome) = outcomes.first() {
                        if outcome.tests_passed {
                            println!("  ✓ PASS\n");
                        } else {
                            println!("  ✗ FAIL: {:?}\n", outcome.error);
                        }
                    }
                }
                Err(e) => {
                    println!("  ✗ ERROR: {}\n", e);
                }
            }
        } else {
            println!("Skipping {} - no golden_patch\n", task.id);
        }
    }

    Ok(())
}

fn cmd_run(
    tasks: &TaskSet,
    backend_name: &str,
    model: Option<&str>,
    attempts: usize,
    filter: Option<&str>,
    dry_run: bool,
    workspace_root: &std::path::Path,
) -> Result<()> {
    let to_run: Vec<_> = if let Some(f) = filter {
        tasks.by_glob(f)
    } else {
        tasks.all().iter().collect()
    };

    let model_id = model.unwrap_or("claude-3-5-sonnet-20241022");

    println!("=== Benchmark Run Configuration ===");
    println!("Backend: {}", backend_name);
    println!("Model: {}", model_id);
    println!("Attempts: {}", attempts);
    println!("Tasks: {}", to_run.len());
    println!();

    if dry_run {
        println!("(dry-run mode - not executing)");
        return Ok(());
    }

    let mut result = EvalResult::new();
    let config = RunConfig {
        workspace_root: workspace_root.to_path_buf(),
        max_attempts: attempts,
        timeout_secs: 120,
    };
    let runner = TaskRunner::new(config);

    let opts = BackendOpts {
        model_id: model_id.to_string(),
        max_tokens: 2048,
        temperature: 0.2,
        timeout_secs: 120,
    };

    for (idx, task) in to_run.iter().enumerate() {
        println!("[{}/{}] Running {}", idx + 1, to_run.len(), task.id);

        let backend: Box<dyn backend::LlmBackend> = match backend_name {
            "local" => Box::new(LocalMlxBackend::new(model_id)),
            "anthropic" => Box::new(AnthropicApiBackend::new()?),
            "debug" => Box::new(StdinDebugBackend),
            _ => {
                return Err(error::BenchError::InvalidConfig(format!(
                    "Unknown backend: {}",
                    backend_name
                )))
            }
        };

        match runner.run_task(task, backend.as_ref(), &opts) {
            Ok(outcomes) => {
                result.outcomes.extend(outcomes);
            }
            Err(e) => {
                eprintln!("  Error: {}", e);
            }
        }
    }

    // Save results
    let results_dir = workspace_root.join("evals/results");
    fs::create_dir_all(&results_dir)?;
    let result_file = results_dir.join(format!("{}_partial.json", result.run_id));
    let json = serde_json::to_string_pretty(&result)?;
    fs::write(result_file, json)?;

    println!("\nRun completed. ID: {}", result.run_id);
    println!("Pass@1: {:.2}%", result.pass_at_1() * 100.0);

    Ok(())
}

fn cmd_report(format_str: &str, latest: bool, file: Option<&std::path::Path>) -> Result<()> {
    let format = match format_str {
        "table" => ReportFormat::Table,
        "json" => ReportFormat::Json,
        "markdown" => ReportFormat::Markdown,
        _ => {
            return Err(error::BenchError::InvalidConfig(format!(
                "Unknown format: {}",
                format_str
            )))
        }
    };

    // Load result
    let result_file = if let Some(f) = file {
        f.to_path_buf()
    } else if latest {
        // Find latest results file
        let results_dir = PathBuf::from("evals/results");
        let entries = fs::read_dir(&results_dir)?;
        let mut latest_file = None;
        let mut latest_time = std::time::SystemTime::UNIX_EPOCH;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(modified) = metadata.modified() {
                        if modified > latest_time {
                            latest_time = modified;
                            latest_file = Some(path);
                        }
                    }
                }
            }
        }

        latest_file.ok_or_else(|| {
            error::BenchError::InvalidConfig("No result files found".to_string())
        })?
    } else {
        return Err(error::BenchError::InvalidConfig(
            "Must specify --file or --latest".to_string(),
        ));
    };

    let json_str = fs::read_to_string(result_file)?;
    let result: EvalResult = serde_json::from_str(&json_str)?;

    let report = Reporter::generate(&result, format);
    println!("{}", report);

    Ok(())
}
