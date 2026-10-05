# DL2026-3-7
Effects of User Prompts on Interactive Image Segmentation.

## Prompt and experiment runner

The implementation owned by Ngô Đức Minh Sơn is documented in
[`docs/son_handoff.md`](docs/son_handoff.md). The complete English experimental
protocol is in [`docs/experimental_setup.md`](docs/experimental_setup.md).

Quick verification from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\validate_inputs.py
.\.venv\Scripts\python.exe scripts\generate_prompts.py
.\.venv\Scripts\python.exe scripts\run_experiments.py --dry-run
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
