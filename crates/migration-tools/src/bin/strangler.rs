use anyhow::Result;
use clap::{Parser, ValueEnum};
use migration_tools::strangler::{Strangler, StranglerConfig};
use serde_json::json;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    Diff,
    Scan,
}

#[derive(Parser)]
#[command(name = "strangler")]
#[command(about = "Report and optionally block changes to configured legacy roots")]
struct Args {
    #[arg(short, long, value_enum, default_value = "diff")]
    mode: Mode,

    #[arg(short, long, default_value = "origin/develop")]
    base: String,

    #[arg(long, default_value = "strangler.json")]
    config: PathBuf,

    #[arg(long)]
    fail_on_legacy: bool,

    #[arg(long)]
    report: bool,

    #[arg(long)]
    json: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = StranglerConfig::load(&args.config)?;
    let repo_root = Strangler::repo_root()?;
    let strangler = Strangler::new(config, repo_root);

    let paths = match args.mode {
        Mode::Diff => strangler.diff_paths(&args.base)?,
        Mode::Scan => strangler.tracked_paths()?,
    };
    let legacy_hits = strangler.classify(&paths);
    let size_report = args.report.then(|| strangler.size_report());

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "mode": format!("{:?}", args.mode).to_lowercase(),
                "checked_paths": paths.len(),
                "legacy_hits": legacy_hits,
                "size_report": size_report,
            }))?
        );
    } else {
        println!("checked_paths={}", paths.len());
        println!("legacy_hits={}", legacy_hits.len());
        for hit in &legacy_hits {
            println!("legacy_path={} root={}", hit.path, hit.legacy_root);
        }
        if let Some(report) = &size_report {
            for item in report {
                println!(
                    "legacy_size root={} bytes={} human={} status={}",
                    item.legacy_root, item.bytes, item.human_size, item.status
                );
            }
        }
    }

    if args.fail_on_legacy && !legacy_hits.is_empty() {
        anyhow::bail!("legacy path changes detected");
    }
    Ok(())
}
