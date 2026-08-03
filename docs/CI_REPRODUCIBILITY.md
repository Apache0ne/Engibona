# CI reproducibility

The repository includes two workflows:

- `tests.yml` for the base unit suite;
- `evidence-max.yml` for the optional Hugging Face dependency set, the official Qwen3-VL text architecture smoke, and a downloadable JSON artifact.

A result should not be treated as independently reproduced until a clean GitHub-hosted runner has:

1. installed from the committed `pyproject.toml`;
2. passed all unit tests;
3. initialized the official `Qwen3VLTextModel` from a tiny configuration;
4. completed a teacher and exact-binary recovery smoke;
5. uploaded `official_qwen3vl_smoke.json`.

Committed result JSON files are evidence records, not substitutes for the clean-run workflow.
