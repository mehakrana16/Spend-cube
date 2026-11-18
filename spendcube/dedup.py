"""Steps 5+6: one supplier_key per real supplier, one item_key per real item."""
import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from .utils import DSU

LEGAL = r"\b(PTY|LTD|LIMITED|INC|LLC|GMBH|CC|CO|CORP|CORPORATION|PLC|BV|SRL|SPA|AG|SAS|SARL|PVT|LLP|SA|SL|SAU|OY|AB|AS|NV)\b"


def norm_name(s):
    s = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper())
    s = re.sub(LEGAL, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def supplier_keys(df, scfg):
    """Name clusters (space-insensitive fuzzy match) linked together through shared vendor ids."""
    scfg = scfg or {}
    thr = scfg.get("name_threshold", 0.88)
    names = df["supplier"].map(lambda x: norm_name(x) if pd.notna(x) else "")
    compact = names.str.replace(" ", "", regex=False)
    dsu = DSU()
    counts = compact[compact != ""].value_counts()
    reps = []
    rep_of = {}
    for name in counts.index:                             # frequent spellings become cluster centres
        hit = next((r for r in reps if abs(len(r) - len(name)) <= max(3, len(r) * .2) and SequenceMatcher(None, r, name).ratio() >= thr), None)
        if hit is None:
            reps.append(name)
            hit = name
        rep_of[name] = hit
    use_id = scfg.get("use_vendor_id", True)
    keys = []
    for cmp, vid in zip(compact, df["vendor_id"]):
        nodes = []
        if cmp:
            nodes.append("N:" + rep_of[cmp])
        if use_id and pd.notna(vid) and str(vid).strip():
            nodes.append("V:" + str(vid).strip())
        for n in nodes[1:]:
            dsu.union(nodes[0], n)
        keys.append(nodes[0] if nodes else None)
    root = [dsu.find(k) if k else None for k in keys]
    uniq = {r: f"SUP-{i:05d}" for i, r in enumerate(sorted({r for r in root if r}))}
    out = pd.Series([uniq.get(r, "SUP-UNKNOWN") for r in root], index=df.index)
    if scfg.get("unmapped_bucket"):
        # Convention seen in the labelled data: names that never appear with a vendor-master id share one bucket.
        has_vid = {dsu.find(n) for n in list(dsu.p) if n.startswith("V:")}
        out = pd.Series(["SUP-UNMAPPED" if (r and r not in has_vid) else o for r, o in zip(root, out)], index=df.index)
    return out


def norm_item(s):
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", str(s).upper())).strip() if pd.notna(s) else ""


def item_keys(df, icfg):
    """Item identity: item code when present (optional config bucket for placeholder codes);
    code-less rows join the closest coded item by text, otherwise cluster among themselves."""
    icfg = icfg or {}
    thr = icfg.get("text_threshold", 0.93)
    txt = df["description"].map(norm_item)
    code = df["item_code"].astype("string").str.strip() if "item_code" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    has_code = code.notna() & (code != "") & icfg.get("use_item_code", True)
    key = pd.Series(pd.NA, index=df.index, dtype="object")
    bucket = pd.Series(False, index=df.index)
    for rule in icfg.get("bucket_rules") or []:              # e.g. catalogue / free-text lines carry placeholder codes
        if rule["field"] in df.columns:
            bucket |= df[rule["field"]].isin(rule["values"])
    key[has_code & bucket] = icfg.get("bucket_key", "ITM-UNCATALOGUED")
    key[has_code & ~bucket] = "C:" + code[has_code & ~bucket]

    # code-less rows: nearest coded (non-bucket) text first
    coded = pd.DataFrame({"t": txt[has_code & ~bucket], "k": key[has_code & ~bucket]})
    coded = coded[coded.t != ""].groupby("t").k.agg(lambda x: x.value_counts().index[0])
    todo = txt[key.isna() & (txt != "")].unique()
    t2k = {}
    if len(todo):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        vec.fit(list(coded.index) + list(todo))
        if len(coded):
            C = vec.transform(coded.index)
            for st in range(0, len(todo), 2000):
                sim = vec.transform(todo[st:st + 2000]) @ C.T
                best, bs = np.asarray(sim.argmax(axis=1)).ravel(), np.asarray(sim.max(axis=1).todense()).ravel()
                for j, (b, v) in enumerate(zip(best, bs)):
                    if v >= thr:
                        t2k[todo[st + j]] = coded.iloc[b]
        rest = [t for t in todo if t not in t2k]
        dsu = DSU()
        if len(rest) > 1:
            X = vec.transform(rest).tocsr()
            for st in range(0, X.shape[0], 1500):
                sim = (X[st:st + 1500] @ X.T).tocoo()
                m = (sim.data >= thr) & (sim.row + st < sim.col)
                for a, b in zip(sim.row[m] + st, sim.col[m]):
                    dsu.union(rest[a], rest[b])
        for t in rest:
            t2k[t] = "T:" + dsu.find(t)
    key[key.isna() & (txt != "")] = txt[key.isna() & (txt != "")].map(t2k)
    uniq = {r: f"ITM-{i:05d}" for i, r in enumerate(sorted({r for r in key.dropna() if not str(r).startswith("ITM-")}))}
    return key.map(lambda r: r if (pd.isna(r) or str(r).startswith("ITM-")) else uniq[r]).fillna("ITM-UNKNOWN")
