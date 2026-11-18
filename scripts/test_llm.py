"""Live LLM test on a small sample (about 8 API calls). Needs GEMINI_API_KEY (free) or GROQ_API_KEY or ANTHROPIC_API_KEY in the terminal.
   python scripts/test_llm.py configs/B.yaml [data_dir] [n_lines]
Compares zero-label accuracy WITHOUT and WITH the LLM on labelled lines, so you can see if the key is worth it."""
import os, sys, time; sys.path.insert(0, ".")
import pandas as pd, yaml
from pathlib import Path
from spendcube.prepare import load_company
from spendcube.match import run_matching
from spendcube.exclude import apply_exclusions
from spendcube.classify import load_taxonomy, classify
from spendcube.evaluate_cls import accuracy
from spendcube.run import _labels

cfg = yaml.safe_load(open(sys.argv[1])); dd = sys.argv[2] if len(sys.argv) > 2 else "data"; n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
from spendcube.llm import resolve
if resolve(cfg["classify"].get("llm", {}) or {}) is None:
    sys.exit("No LLM key found in this terminal (GEMINI_API_KEY / GROQ_API_KEY / ANTHROPIC_API_KEY). Set one first.")
df = apply_exclusions(run_matching(load_company(cfg, dd), cfg.get("match")), cfg).set_index("record_id", drop=False)
paths, depth = load_taxonomy(Path(dd) / cfg["taxonomy"]["file"], cfg["taxonomy"]["level_columns"])
lab = _labels(cfg, dd, paths); lab.index = lab.index.astype(str)
j = df[df.include == 1].join(lab[["leaf"]], how="inner").dropna(subset=["leaf"])
j = j[j.description.notna()].sample(n, random_state=0)
cols = cfg["labels"]["level_columns"]; raw = lab.loc[j.index, cols]
truth = pd.DataFrame({f"l{i+1}": raw[c].fillna("").values for i, c in enumerate(cols)}, index=j.index)
off = dict(cfg["classify"], llm={"enabled": False})
on = dict(cfg["classify"], llm=dict(cfg["classify"].get("llm", {}) or {}, enabled=True))
if os.path.exists(".llm_cache.json"): os.remove(".llm_cache.json")
a = classify(j, paths, depth, off, labels=None)
t = time.time(); b = classify(j, paths, depth, on, labels=None); sec = time.time() - t
print(f"\nSample: {n} labelled lines, labels hidden from the classifier")
print("Local model only :", accuracy(a, truth, depth))
print("With LLM         :", accuracy(b, truth, depth), f"(method={b.method.iloc[0]}, {sec:.0f}s)")
print("If 'With LLM' is clearly higher the key is working. If method has no '+llm' the API calls failed (see [llm] lines above).")