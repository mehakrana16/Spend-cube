"""Step 1+3: ingest by config, clean amounts / tax / currency / dates. No company-specific code here."""
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import parse_number, parse_rate, parse_dates


def _col(df, spec):
    """Resolve a config column spec: None -> NaN, str -> column, list -> blank-skipping join."""
    if spec is None:
        return pd.Series(np.nan, index=df.index, dtype="object")
    if isinstance(spec, list):
        cols = [c for c in spec if c in df.columns]
        if not cols:
            return pd.Series(np.nan, index=df.index, dtype="object")
        joined = df[cols].fillna("").astype(str).agg(" | ".join, axis=1)
        joined = joined.str.replace(r"(\s*\|\s*)+$", "", regex=True).str.replace(r"^(\s*\|\s*)+", "", regex=True)
        return joined.replace("", np.nan)
    return df[spec] if spec in df.columns else pd.Series(np.nan, index=df.index, dtype="object")


def _fx_lookup(fx, cur, month_ord, cfg):
    """Rate (units per reporting ccy) for each (currency, month); nearest available month if gap."""
    f = fx.rename(columns={cfg["currency_col"]: "cur", cfg["month_col"]: "m", cfg["rate_col"]: "r"})
    f["r"] = f["r"].map(parse_number)
    f["mo"] = pd.PeriodIndex(f["m"].astype(str).str[:7], freq="M").astype("int64")
    out = pd.Series(np.nan, index=cur.index)
    for c, g in f.dropna(subset=["r"]).groupby("cur"):
        g = g.sort_values("mo")
        idx = (cur == c).values
        if not idx.any():
            continue
        mo = month_ord[idx].values
        pos = np.searchsorted(g["mo"].values, mo)
        left = np.clip(pos - 1, 0, len(g) - 1)
        right = np.clip(pos, 0, len(g) - 1)
        pick = np.where(np.abs(g["mo"].values[left] - mo) <= np.abs(g["mo"].values[right] - mo), left, right)
        out[idx] = g["r"].values[pick]
    return out


def load_source(cfg, name, src, data_dir):
    df = pd.read_csv(Path(data_dir) / src["file"], dtype=str, encoding=src.get("encoding", "utf-8"))
    c = src["columns"]
    out = pd.DataFrame(index=df.index)
    out["record_id"] = src.get("id_prefix", "") + df[src["id_column"]].astype(str)
    out["system"] = name
    for k in ["supplier", "vendor_id", "item_code", "po", "po_item", "invoice_number", "status_raw", "description"]:
        out[k] = _col(df, c.get({"status_raw": "status"}.get(k, k)))
    out["quantity"] = _col(df, c.get("quantity")).map(parse_number)

    # --- dates
    out["date"], out["date_status"] = parse_dates(_col(df, c.get("date")), src.get("date_formats") or cfg.get("date_formats"))
    lo, hi = (pd.Timestamp(x) for x in cfg.get("date_valid_range", ["1990-01-01", "2100-01-01"]))
    bad = out["date"].notna() & ((out["date"] < lo) | (out["date"] > hi))
    out.loc[bad, "date_status"] = "out_of_range"
    fx_date_col = (cfg.get("fx") or {}).get("date_column") or c.get("date")
    fxd, _ = parse_dates(_col(df, fx_date_col), src.get("date_formats") or cfg.get("date_formats"))
    fxd = fxd.fillna(out["date"])

    # --- amount (net of tax)
    a = src["amount"]
    amt = df[a["column"]].map(parse_number)
    out["amount_status"] = np.where(amt.isna(), "unparsed", "ok")
    if a.get("gross"):
        vat = df[a["vat_column"]].map(parse_rate) if a.get("vat_column") in df.columns else pd.Series(np.nan, index=df.index)
        out["vat_rate"] = vat.fillna(a.get("vat_default", 0.0))
        amt = amt / (1 + out["vat_rate"])
    out["amount_native"] = amt

    # --- currency -> reporting currency
    rep = cfg["reporting_currency"].upper()
    cc = src.get("currency", {})
    cur = df[cc["column"]].str.strip().str.upper() if cc.get("column") in df.columns else pd.Series(cc.get("fixed", rep), index=df.index)
    cur = cur.fillna(rep)
    rate = df[cc["rate_column"]].map(parse_number) if cc.get("rate_column") in df.columns else pd.Series(np.nan, index=df.index)
    out["currency"] = cur
    spend = pd.Series(np.nan, index=df.index)
    spend[cur == rep] = amt[cur == rep]                    # already in reporting currency: ignore any stray rate
    use_rate = (cur != rep) & (rate > 0)
    spend[use_rate] = amt[use_rate] / rate[use_rate]
    need_fx = (cur != rep) & ~(rate > 0)
    out["fx_source"] = np.where(cur == rep, "none", np.where(use_rate, "line_rate", "fx_table"))
    if need_fx.any() and cfg.get("fx"):
        fx = pd.read_csv(Path(data_dir) / cfg["fx"]["file"], dtype=str)
        mo = pd.PeriodIndex(fxd.dt.to_period("M").fillna(pd.Period("2024-01", "M")), freq="M").astype("int64")
        r = _fx_lookup(fx, cur, pd.Series(mo, index=df.index), cfg["fx"])
        spend[need_fx] = amt[need_fx] / r[need_fx]
    out["fx_source"] = np.where(spend.isna() & (out["fx_source"] != "none"), "missing", out["fx_source"])
    out["spend_raw"] = spend

    # --- status
    st = out["status_raw"].astype("string").str.strip().str.lower()
    sc = src.get("status")
    if sc and c.get("status"):
        inc = [s.lower() for s in sc["include"]]
        canc = [s.lower() for s in sc.get("cancelled", [])]
        out["status_class"] = np.where(st.isin(inc), "ok", np.where(st.isin(canc), "cancelled", "not_approved"))
    else:
        out["status_class"] = "ok"

    # --- keep raw columns so exclusion rules / classifier features can reference any original column
    for k in df.columns:
        out["raw_" + k] = df[k]
    return out


def load_company(cfg, data_dir):
    frames = [load_source(cfg, n, s, data_dir) for n, s in cfg["sources"].items()]
    return pd.concat(frames, ignore_index=True)
