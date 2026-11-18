"""Step 4: decide what counts as spend. Every excluded record gets one reason code."""
import warnings
warnings.filterwarnings("ignore", message="This pattern is interpreted")
import re
import numpy as np
import pandas as pd


def apply_exclusions(df, cfg):
    reason = pd.Series("", index=df.index)
    reason[df.status_class == "cancelled"] = "CANCELLED"
    reason[(df.status_class == "not_approved") & (reason == "")] = "NOT_APPROVED"
    reason[(df.dup_reason != "") & (reason == "")] = "DUPLICATE"
    for rule in cfg.get("exclusions") or []:
        hit = pd.Series(False, index=df.index)
        for cond in rule["any"]:
            if cond["field"] in df.columns:
                hit |= df[cond["field"]].fillna("").astype(str).str.contains(cond["regex"], flags=re.I, regex=True)
        reason[hit & (reason == "")] = rule["reason"]
    df = df.copy()
    df["exclusion_reason"] = reason
    df["include"] = (reason == "").astype(int)
    df["spend"] = np.where(df.include == 1, df.spend_raw.fillna(0.0), 0.0)
    return df
