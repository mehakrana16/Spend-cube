"""Small generic helpers: number/date parsing, union-find, pairwise clustering metrics."""
import re
import numpy as np
import pandas as pd


def parse_number(x):
    """'5,266.63' / '(120.5)' / '1.234,50' / '12-' -> float; unparseable -> NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.endswith("-"):
        neg, s = True, s[:-1]
    s = re.sub(r"[^\d,.\-]", "", s)
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        s = s.replace(",", "") if (len(parts) > 1 and len(parts[-1]) == 3) else s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v


def parse_rate(x):
    """VAT/tax rate: '15', '0.15', '15%' -> 0.15 ; blank -> NaN."""
    v = parse_number(str(x).replace("%", "")) if pd.notna(x) else np.nan
    if pd.isna(v):
        return np.nan
    return v / 100.0 if v > 1 else v


def parse_dates(series, formats=None):
    """Try each configured format, then a lenient day-first fallback. Returns (dates, status)."""
    s = series.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for f in (formats or []):
        m = out.isna() & s.notna() & (s != "")
        if not m.any():
            break
        out[m] = pd.to_datetime(s[m], format=f, errors="coerce")
    m = out.isna() & s.notna() & (s != "")
    if m.any():
        out[m] = pd.to_datetime(s[m], errors="coerce", dayfirst=True)
    status = np.where(s.isna() | (s == ""), "missing", np.where(out.isna(), "unparsed", "ok"))
    return out, pd.Series(status, index=s.index)


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def pairwise_scores(pred, truth):
    """Pairwise precision / recall / F1 of a clustering vs. golden keys."""
    df = pd.DataFrame({"p": list(pred), "t": list(truth)}).dropna()
    c2 = lambda n: n * (n - 1) / 2
    tp = df.groupby(["p", "t"]).size().map(c2).sum()
    pp = df.groupby("p").size().map(c2).sum()
    tt = df.groupby("t").size().map(c2).sum()
    prec, rec = (tp / pp if pp else 1.0), (tp / tt if tt else 1.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
