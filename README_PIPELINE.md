# Ddarengi Replay Pipeline

Run a simple replay of rental history to compute stockout/full metrics.

Quickstart:

1. Create a virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the replay:

```bash
python run_replay.py
```

The script reads CSVs from `data/ddarengi/` and uses `config/default.yaml` for parameters.
