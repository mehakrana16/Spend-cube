"""Supervised accuracy with an honest hold-out: python scripts/holdout.py configs/B.yaml [data_dir]
Trains on 80% of the labelled lines, scores the other 20% (random split) and a split by unseen description text."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, yaml
from pathlib import Path
from spendcube.prepare import load_company
from spendcube.match import run_matching
from spendcube.exclude import apply_exclusions
from spendcube.classify import load_taxonomy, classify
from spendcube.evaluate_cls import accuracy
from spendcube.run import _labels

cfg = yaml.safe_load(open(sys.argv[1])); dd = sys.argv[2] if len(sys.argv) > 2 else "data"
df = apply_exclusions(run_matching(load_company(cfg, dd), cfg.get("match")), cfg).set_index("record_id", drop=False)
paths, depth = load_taxonomy(Path(dd) / cfg["taxonomy"]["file"], cfg["taxonomy"]["level_columns"])
lab = _labels(cfg, dd, paths); lab.index = lab.index.astype(str)
j = df[df.include == 1].join(lab[["leaf"]], how="inner").dropna(subset=["leaf"])
cols = cfg["labels"]["level_columns"]; raw = lab.loc[j.index, cols]
truth = pd.DataFrame({f"l{i+1}": raw[c].fillna("").values for i, c in enumerate(cols)}, index=j.index)
rng = np.random.RandomState(0)
txt = j.description.fillna("")
for tag, te in [("random 20% (new lines of known items)", rng.rand(len(j)) < .2),
                ("unseen descriptions (new items)", txt.isin(set(rng.choice(txt.unique(), int(txt.nunique() * .2), replace=False))).values)]:
    train = j[~te][["leaf"]].astype(int)
    pred = classify(j, paths, depth, cfg.get("classify"), labels=train)
    print(tag, "n_test =", int(te.sum()), accuracy(pred[te], truth[te], depth), "| spend-weighted", accuracy(pred[te], truth[te], depth, j.spend.abs().values[te]))
