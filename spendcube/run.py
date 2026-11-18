"""One command, end to end:
   python -m spendcube.run --config configs/A.yaml --data-dir data --out outputs --name candidate
Add --hide-labels to ignore any labels file (zero-label mode); --evaluate to print accuracy vs. labels."""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .prepare import load_company
from .match import run_matching
from .exclude import apply_exclusions
from .dedup import supplier_keys, item_keys
from .classify import load_taxonomy, classify
from .evaluate_cls import accuracy
from .utils import pairwise_scores

OUT_COLS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
            "l1", "l2", "l3", "l4", "confidence"]
EXCL_CONF = {"CANCELLED": 0.99, "NOT_APPROVED": 0.99, "DUPLICATE": 0.9, "NON_PROCUREMENT": 0.85}


def _labels(cfg, data_dir, paths):
    lc = cfg.get("labels")
    if not lc or not (Path(data_dir) / lc["file"]).exists():
        return None
    lab = pd.read_csv(Path(data_dir) / lc["file"], dtype=str).set_index(lc["id_column"])
    pm = {p: i for i, p in enumerate(paths)}
    cols = lc["level_columns"]
    lab["leaf"] = [pm.get(tuple(str(x).strip() for x in row if pd.notna(x) and str(x).strip())) for row in lab[cols].values]
    return lab


def run(config, data_dir, out_dir, name, hide_labels=False, evaluate=False):
    t0 = time.time()
    cfg = yaml.safe_load(open(config))
    df = load_company(cfg, data_dir)
    df = run_matching(df, cfg.get("match"))
    df = apply_exclusions(df, cfg)
    df["supplier_key"] = supplier_keys(df, cfg.get("supplier"))
    df["item_key"] = item_keys(df, cfg.get("item"))

    paths, depth = load_taxonomy(Path(data_dir) / cfg["taxonomy"]["file"], cfg["taxonomy"]["level_columns"])
    lab = None if hide_labels else _labels(cfg, data_dir, paths)
    train = None
    if lab is not None:
        train = lab[["leaf"]].dropna().astype(int)
        train.index = train.index.astype(str)
    inc = df[df.include == 1].set_index("record_id", drop=False)
    cls = classify(inc, paths, depth, cfg.get("classify"), labels=train.reindex(inc.index).dropna().astype(int) if train is not None else None)

    out = df.set_index("record_id", drop=False)
    for d in range(1, 5):
        out[f"l{d}"] = ""
    for d in range(1, depth + 1):
        out.loc[cls.index, f"l{d}"] = cls[f"l{d}"]
    out["confidence"] = out["exclusion_reason"].map(EXCL_CONF).fillna(0.0)
    out.loc[cls.index, "confidence"] = cls["confidence"]
    out["spend"] = out["spend"].round(2)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(out_dir) / f"{name}__{cfg['name']}.csv"
    out[OUT_COLS].to_csv(fp, index=False)

    # ---- run report
    print(f"\n=== {cfg['name']}: {len(out):,} records -> {fp}  ({time.time()-t0:.0f}s, classifier: {cls['method'].iloc[0] if len(cls) else 'n/a'})")
    print("included:", int(out.include.sum()), "| total spend", f"{out.spend.sum():,.0f}", cfg["reporting_currency"])
    print("exclusions:", out.exclusion_reason.replace("", "(included)").value_counts().to_dict())
    print("flags: date", df.date_status.value_counts().to_dict(), "| fx", df.fx_source.value_counts().to_dict(),
          "| amount unparsed", int((df.amount_status == "unparsed").sum()))
    if evaluate:
        evaluate_run(cfg, df, out, paths, depth, data_dir, hide_labels)
    return out


def evaluate_run(cfg, df, out, paths, depth, data_dir, hide):
    lc = cfg.get("labels")
    lab = pd.read_csv(Path(data_dir) / lc["file"], dtype=str).set_index(lc["id_column"]) if lc else None
    if lab is None:
        return
    j = out.join(lab, how="inner", rsuffix="_lab")
    print(f"--- evaluation on {len(j):,} labelled records (labels hidden from classifier: {hide})")
    for k in ("supplier_key", "item_key"):
        if lc.get(k) and lc[k] in lab.columns:
            tcol = lc[k] + "_lab" if lc[k] in out.columns else lc[k]
            jj = j.dropna(subset=[tcol])
            print(f"{k} pairwise:", pairwise_scores(jj[k], jj[tcol]))
    if "include" in lab.columns:
        print("include agreement with seed:", round(float((j["include"].astype(int) == j["include_lab"].astype(int)).mean()), 4))
    cols = lc["level_columns"]
    jj = j[(j.include == 1) & j[cols[0]].notna()]
    truth = pd.DataFrame({f"l{i+1}": jj[c].fillna("").values for i, c in enumerate(cols)}, index=jj.index)
    print("classification acc (unweighted):", accuracy(jj, truth, len(cols)))
    print("classification acc (spend-weighted):", accuracy(jj, truth, len(cols), jj.spend.abs().values))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--name", default="candidate")
    ap.add_argument("--hide-labels", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    run(a.config, a.data_dir, a.out, a.name, a.hide_labels, a.evaluate)
