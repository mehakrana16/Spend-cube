"""Step 2: count once. Match records of a secondary system (invoice tool) to a primary system (ERP)."""
import re
from difflib import SequenceMatcher
import numpy as np
import pandas as pd


def _norm_text(s):
    return re.sub(r"[^A-Z0-9]+", " ", str(s).upper()).strip() if pd.notna(s) else ""


def _digits(s):
    runs = re.findall(r"\d+", str(s)) if pd.notna(s) else []
    return max(runs, key=len) if runs else ""


def run_matching(df, mcfg):
    """Adds columns: dup_reason ('' | 'SECONDARY_REPEAT' | 'MATCHED_PRIMARY'), matched_to."""
    df = df.copy()
    df["dup_reason"], df["matched_to"] = "", ""
    if not mcfg:
        return df
    P, S = mcfg["primary"], mcfg["secondary"]
    po_re, poi_re = re.compile(mcfg.get("po_regex", r"(\d{6,})")), re.compile(mcfg.get("po_item_regex", r"/(\d+)"))
    tol = mcfg.get("amount_tolerance", 0.001)
    s_idx = df.index[(df.system == S) & (df.status_class == "ok")]
    p_idx = df.index[(df.system == P) & (df.status_class == "ok")]

    ref = df["po"].astype("string")
    df["po_n"] = ref.str.extract(po_re, expand=False)
    df["poi_n"] = pd.to_numeric(ref.str.extract(poi_re, expand=False), errors="coerce")
    df.loc[df.system == P, "poi_n"] = pd.to_numeric(df.loc[df.system == P, "po_item"], errors="coerce")
    df["txt_n"] = df["description"].map(_norm_text)
    df["net"] = df["spend_raw"]

    # (a) the same invoice line exported twice inside the secondary system
    if mcfg.get("secondary_dedupe", True):
        key = (df.loc[s_idx, "invoice_number"].map(_digits) + "|" + df.loc[s_idx, "po_n"].fillna("") + "|"
               + df.loc[s_idx, "net"].round(2).astype(str) + "|" + df.loc[s_idx, "txt_n"] + "|" + df.loc[s_idx, "quantity"].astype(str))
        rep = key.duplicated(keep="first")
        df.loc[rep[rep].index, "dup_reason"] = "SECONDARY_REPEAT"
        df.loc[rep[rep].index, "matched_to"] = key[rep].map(key[~rep].reset_index().drop_duplicates(0).set_index(0)["index"].map(df["record_id"]))
        s_idx = s_idx.difference(rep[rep].index)

    # (b) secondary line -> primary line. One-to-one, closest amount wins.
    used = set()
    by_po = df.loc[p_idx].groupby("po_n").groups
    def amount_ok(a, b):
        return pd.notna(a) and pd.notna(b) and abs(a - b) <= tol * max(abs(b), 1.0)

    def qty_eq(a, b):
        return pd.isna(a) or pd.isna(b) or abs(a - b) < 1e-6

    def take(si, cands, require_text=False, loose=False):
        best, best_d = None, None
        for pi in cands:
            if pi in used:
                continue
            if loose:      # same PO + same quantity + near-identical text, amount differs (price variance)
                if not qty_eq(df.at[si, "quantity"], df.at[pi, "quantity"]) or pd.isna(df.at[si, "quantity"]):
                    continue
                if SequenceMatcher(None, df.at[si, "txt_n"], df.at[pi, "txt_n"]).ratio() < mcfg.get("loose_text_similarity", 0.85):
                    continue
            elif not amount_ok(df.at[si, "net"], df.at[pi, "net"]):
                continue
            if require_text and SequenceMatcher(None, df.at[si, "txt_n"], df.at[pi, "txt_n"]).ratio() < mcfg.get("fallback_text_similarity", 0.6):
                continue
            d = abs(df.at[si, "net"] - df.at[pi, "net"]) + (0 if pd.isna(df.at[si, "poi_n"]) or df.at[si, "poi_n"] == df.at[pi, "poi_n"] else 1e-6)
            if best is None or d < best_d:
                best, best_d = pi, d
        if best is not None:
            used.add(best)
            df.at[si, "dup_reason"], df.at[si, "matched_to"] = "MATCHED_PRIMARY", df.at[best, "record_id"]
            return True
        return False

    left = []
    for si in s_idx:
        po = df.at[si, "po_n"]
        if pd.notna(po) and po in by_po and take(si, by_po[po]):
            continue
        left.append(si)
    if mcfg.get("loose_po_pass", True):
        still = []
        for si in left:
            po = df.at[si, "po_n"]
            if pd.notna(po) and po in by_po and take(si, by_po[po], loose=True):
                continue
            still.append(si)
        left = still

    # (c) fallback: no PO (or PO not in primary): same net amount + same quantity + similar text
    if mcfg.get("fallback_without_po", True):
        by_amt = {}
        for pi in p_idx:
            if pi not in used and pd.notna(df.at[pi, "net"]):
                by_amt.setdefault(round(df.at[pi, "net"], 1), []).append(pi)
        for si in left:
            a = df.at[si, "net"]
            if pd.isna(a):
                continue
            cands = [p for k in (round(a, 1) - 0.1, round(a, 1), round(a, 1) + 0.1) for p in by_amt.get(round(k, 1), [])]
            cands = [p for p in cands if (pd.isna(df.at[si, "quantity"]) or pd.isna(df.at[p, "quantity"]) or abs(df.at[si, "quantity"] - df.at[p, "quantity"]) < 1e-6)]
            take(si, cands, require_text=True)
    return df.drop(columns=["po_n", "poi_n", "txt_n", "net"])
