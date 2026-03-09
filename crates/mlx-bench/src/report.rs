use crate::runner::EvalResult;
use serde_json::json;
use std::collections::HashMap;

#[derive(Debug)]
pub enum ReportFormat {
    Table,
    Json,
    Markdown,
}

pub struct Reporter;

impl Reporter {
    pub fn generate(result: &EvalResult, format: ReportFormat) -> String {
        match format {
            ReportFormat::Table => Self::table_format(result),
            ReportFormat::Json => Self::json_format(result),
            ReportFormat::Markdown => Self::markdown_format(result),
        }
    }

    fn table_format(result: &EvalResult) -> String {
        let mut output = String::new();
        output.push_str("SWE-Bench Results\n");
        output.push_str(&format!("Run ID: {}\n", result.run_id));
        output.push_str(&format!("Created: {}\n\n", result.created_at));

        // Aggregate by task
        let mut task_map: HashMap<String, Vec<_>> = HashMap::new();
        for outcome in &result.outcomes {
            task_map
                .entry(outcome.task_id.clone())
                .or_default()
                .push(outcome);
        }

        // Summary metrics
        output.push_str("=== SUMMARY ===\n");
        output.push_str(&format!("Pass@1: {:.2}%\n", result.pass_at_1() * 100.0));
        output.push_str(&format!("Pass@3: {:.2}%\n", result.pass_at_k(3) * 100.0));
        output.push_str(&format!("Pass@5: {:.2}%\n", result.pass_at_k(5) * 100.0));
        output.push_str(&format!("Compilation Rate: {:.2}%\n\n", result.compilation_rate() * 100.0));

        // Task details
        output.push_str("=== TASK RESULTS ===\n");
        output.push_str("Task ID                  | Attempt | Build | Tests | Status\n");
        output.push_str(&"-".repeat(60));
        output.push('\n');

        for (task_id, outcomes) in task_map.iter() {
            for outcome in outcomes {
                let status = if outcome.tests_passed {
                    "✓ PASS"
                } else if let Some(ref err) = outcome.error {
                    &format!("✗ {}", err.split('\n').next().unwrap_or("ERROR"))
                } else {
                    "✗ FAIL"
                };

                output.push_str(&format!(
                    "{:<24} | {:<7} | {:<5} | {:<5} | {}\n",
                    task_id,
                    outcome.attempt,
                    if outcome.build_success { "✓" } else { "✗" },
                    if outcome.tests_passed { "✓" } else { "✗" },
                    status
                ));
            }
        }

        output
    }

    fn json_format(result: &EvalResult) -> String {
        let json_obj = json!({
            "run_id": result.run_id,
            "created_at": result.created_at,
            "pass_at_1": result.pass_at_1(),
            "pass_at_3": result.pass_at_k(3),
            "pass_at_5": result.pass_at_k(5),
            "compilation_rate": result.compilation_rate(),
            "outcomes": result.outcomes,
        });

        serde_json::to_string_pretty(&json_obj).unwrap_or_default()
    }

    fn markdown_format(result: &EvalResult) -> String {
        let mut output = String::new();
        output.push_str("# SWE-Bench Results\n\n");
        output.push_str(&format!("**Run ID:** {}\n", result.run_id));
        output.push_str(&format!("**Created:** {}\n\n", result.created_at));

        // Summary table
        output.push_str("## Summary Metrics\n\n");
        output.push_str("| Metric | Value |\n");
        output.push_str("|--------|-------|\n");
        output.push_str(&format!("| Pass@1 | {:.2}% |\n", result.pass_at_1() * 100.0));
        output.push_str(&format!("| Pass@3 | {:.2}% |\n", result.pass_at_k(3) * 100.0));
        output.push_str(&format!("| Pass@5 | {:.2}% |\n", result.pass_at_k(5) * 100.0));
        output.push_str(&format!(
            "| Compilation Rate | {:.2}% |\n",
            result.compilation_rate() * 100.0
        ));
        output.push_str("\n");

        // Aggregate by task
        let mut task_map: HashMap<String, Vec<_>> = HashMap::new();
        for outcome in &result.outcomes {
            task_map
                .entry(outcome.task_id.clone())
                .or_default()
                .push(outcome);
        }

        output.push_str("## Task Results\n\n");
        output.push_str("| Task ID | Attempt | Build | Tests | Status |\n");
        output.push_str("|---------|---------|-------|-------|--------|\n");

        for (task_id, outcomes) in task_map.iter() {
            for outcome in outcomes {
                let status = if outcome.tests_passed {
                    "✓ PASS"
                } else if let Some(ref err) = outcome.error {
                    &format!("✗ {}", err.split('\n').next().unwrap_or("ERROR"))
                } else {
                    "✗ FAIL"
                };

                output.push_str(&format!(
                    "| {} | {} | {} | {} | {} |\n",
                    task_id,
                    outcome.attempt,
                    if outcome.build_success { "✓" } else { "✗" },
                    if outcome.tests_passed { "✓" } else { "✗" },
                    status
                ));
            }
        }

        output
    }
}
