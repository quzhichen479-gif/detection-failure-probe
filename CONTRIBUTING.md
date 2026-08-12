# Contributing

Thanks for helping make Detection Failure Probe more useful and trustworthy.

## Before opening a change

- Use an issue for behavior changes large enough to need agreement on scope.
- Keep the tool local-first and CPU-capable.
- Do not add telemetry, remote uploads, model downloads, unsafe deserialization, or dataset mutation
  as incidental features.
- Do not commit private datasets, predictions, reviewer notes, credentials, or generated `runs/`.

## Development setup

```bash
git clone https://github.com/quzhichen479-gif/detection-failure-probe.git
cd detection-failure-probe
python -m venv .venv
# activate the virtual environment for your shell
pip install -e ".[dev]"
python scripts/generate_demo.py
pytest
ruff check .
```

## Pull requests

1. Add focused tests for new or changed behavior.
2. Run the full test and lint commands.
3. Update README input/output contracts when formats or metrics change.
4. Explain metric semantics and limitations; do not imply that diagnostics predict model quality.
5. Keep generated artifacts and unrelated formatting changes out of the commit.

Bug reports should include the command, Python version, platform, expected behavior, and a minimal
synthetic reproduction when possible. Never attach a dataset you do not have permission to share.
