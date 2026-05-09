use anyhow::Result;
use clap::Parser;
use migration_tools::openapi;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "openapi-extractor")]
#[command(about = "Discover Rust HTTP services and emit OpenAPI template artifacts")]
struct Args {
    #[arg(short, long)]
    root: Option<PathBuf>,

    #[arg(short, long, default_value = "target/migration-tools/openapi")]
    output_dir: PathBuf,

    #[arg(long)]
    fail_on_empty: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = args.root.unwrap_or(std::env::current_dir()?);
    let services = openapi::discover_services(&root)?;
    let report = openapi::discovery_report(services);
    openapi::write_outputs(&root, &args.output_dir, &report)?;

    println!("discovered_services={}", report.service_count);
    println!(
        "discovery_report={}",
        args.output_dir.join("openapi-discovery.json").display()
    );

    if args.fail_on_empty && report.service_count == 0 {
        anyhow::bail!("no Rust HTTP services discovered");
    }
    Ok(())
}
