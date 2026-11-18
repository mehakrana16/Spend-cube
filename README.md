# Spend cube pipeline (Cantill.ai Data & ML task, Round 1)

Raw purchase data + category tree -> one cleaned, de-duplicated, classified spend row per input record.
Everything company specific lives in `configs/*.yaml`; a new company needs a new config, not new code.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pandas, numpy, scikit-learn, pyyaml
mkdir data                               # put the task CSVs here (git-ignored, never committed)
```

## The one command
```bash
python -m spendcube.run --config configs/A.yaml --data-dir data --out outputs --name <yourname>
python -m spendcube.run --config configs/B.yaml --data-dir data --out outputs --name <yourname>
```
Writes `outputs/<yourname>__A.csv` / `__B.csv` (format of `output_template.csv`) and prints a run report.
A (65k records, two systems) takes about 1 minute, B (41k) about 30 seconds on a laptop CPU.

Useful flags: `--hide-labels` (zero-label mode, ignore any labels file), `--evaluate` (accuracy vs. labels),
`python scripts/holdout.py configs/A.yaml` (honest supervised hold-out).

## New company (e.g. blind-test C)
Copy `configs/B.yaml`, change file names, column names, date formats, status values, exclusion regexes and the
taxonomy `level_columns` (up to 4 levels). Delete the `labels:` block if there are no labels. Run the same command.

## Check that it works
```bash
python scripts/check_output.py configs/A.yaml outputs/<yourname>__A.csv      # 11 PASS/FAIL rules + example rows
python -m spendcube.run --config configs/A.yaml --hide-labels --evaluate      # accuracy with labels hidden
python scripts/holdout.py configs/A.yaml                                      # honest supervised hold-out
```

## Optional LLM step
If `ANTHROPIC_API_KEY` is set, unlabelled classification (e.g. company C) automatically asks an LLM
(default `claude-haiku-5-5`, set `classify.llm.model`) to pick the leaf for each unique description; answers are cached in
`.llm_cache.json`. Without a key (or if a call fails) the local zero-label model is used, so the tool always runs.
Never commit your key; set it only in your terminal: `$env:ANTHROPIC_API_KEY="..."` (PowerShell).

## Layout
`spendcube/prepare.py` ingest + amounts/tax/FX/dates · `match.py` count-once matching · `exclude.py` reason codes ·
`dedup.py` supplier/item keys · `classify.py` taxonomy classifier · `llm.py` optional LLM · `run.py` CLI.
