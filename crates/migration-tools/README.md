# migration-tools

Repository-local CI tools for tracking API discovery and strangler-pattern migration progress.

## Binaries

- `openapi-extractor`: scans Rust workspace service directories for crates that directly depend on supported HTTP frameworks (`axum`, `actix-web`, `rocket`) and writes OpenAPI template artifacts.
- `strangler`: scans configured legacy roots, reports size metrics, and blocks pull requests that change legacy paths.

## Examples

```bash
cargo run -p migration-tools --bin openapi-extractor -- --root . --output-dir target/migration-tools/openapi
cargo run -p migration-tools --bin strangler -- --mode scan --report --json
cargo run -p migration-tools --bin strangler -- --mode diff --base origin/develop --fail-on-legacy
```
