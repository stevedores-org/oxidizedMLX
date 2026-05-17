# Contributing

Keep the migration tools deterministic and CI-friendly:

- Prefer structured parsers over substring matching for repository metadata.
- Return errors when CI guard inputs are missing or invalid.
- Add fixture-based tests for new discovery or strangler behavior.
- Keep emitted JSON stable enough for artifact diffing and dashboards.
