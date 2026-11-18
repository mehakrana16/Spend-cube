"""Sanity-check an output file against the brief.
   python scripts/check_output.py configs/A.yaml outputs/Mehak__A.csv [data_dir]
Prints PASS/FAIL for each rule and shows real example rows (raw input -> cleaned result) so you can see the work."""
import sys; sys.path.insert(0, ".")
import pandas as pd, yaml
from pathlib import Path
from spendcube.prepare import load_company
from spendcube.classify import load_taxonomy
from spendcube.match import run_matching

cfg = yaml.safe_load(open(sys.argv[1])); out = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False)
dd = sys.argv[3] if len(sys.argv) > 3 else "data"
raw = run_matching(load_company(cfg, dd), cfg.get("match"))
paths, depth = load_taxonomy(Path(dd) / cfg["taxonomy"]["file"], cfg["taxonomy"]["level_columns"])
tree_names = {n for p in paths for n in p}
ok_all = True
def check(name, cond, detail=""):
    global ok_all; ok_all &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))

o = out.copy(); o["include"] = o["include"].astype(int); o["spend"] = o["spend"].astype(float); o["confidence"] = o["confidence"].astype(float)
print(f"\n=== Checks for {sys.argv[2]} ===")
check("1 row per input record", len(o) == len(raw), f"{len(o)} output vs {len(raw)} input")
check("record_id unique and same ids as input", o.record_id.is_unique and set(o.record_id) == set(raw.record_id))
check("columns as in template", list(o.columns) == ["record_id","include","exclusion_reason","supplier_key","item_key","spend","l1","l2","l3","l4","confidence"])
inc, exc = o[o.include == 1], o[o.include == 0]
check("excluded rows have a valid reason code", exc.exclusion_reason.isin(["CANCELLED","NOT_APPROVED","NON_PROCUREMENT","DUPLICATE"]).all(), str(exc.exclusion_reason.value_counts().to_dict()))
check("included rows have blank reason", (inc.exclusion_reason == "").all())
check("excluded rows have spend 0", (exc.spend == 0).all())
check("every included row has l1 and a valid tree name", (inc.l1 != "").all() and inc.l1.isin(tree_names).all())
check("l1..l4 names exist in the tree file", all(inc[c][inc[c] != ""].isin(tree_names).all() for c in ["l1","l2","l3","l4"]))
check("confidence within 0..1", o.confidence.between(0, 1).all())
check("no empty supplier_key / item_key", (o.supplier_key != "").all() and (o.item_key != "").all())
m = raw.set_index("record_id").join(o.set_index("record_id")[["include","spend"]].rename(columns={"spend": "out_spend"}))
check("cancelled / not-approved never counted", ((m.status_class == "ok") | (m.include == 0)).all())
print(f"\nTotals: included {len(inc):,} of {len(o):,} | total spend {o.spend.sum():,.0f} {cfg['reporting_currency']}")

print("\n--- Example 1: tax / currency cleaning (raw amount -> spend in reporting currency)")
cols = ["system", "amount_native", "vat_rate", "currency", "fx_source", "out_spend"] if "vat_rate" in m else ["system", "amount_native", "currency", "fx_source", "out_spend"]
shown = []
for key, grp in m[m.include == 1].groupby(["system", "fx_source"]):
    shown.append(grp[cols].dropna(subset=["amount_native"]).head(2))
print(pd.concat(shown).to_string())
if "matched_to" in m:
    d = m[m.dup_reason != ""].reset_index()
    if len(d):
        print("\n--- Example 2: records excluded as DUPLICATE (should point to the counted record)")
        print(d[["record_id", "dup_reason", "matched_to", "out_spend"]].head(4).to_string(index=False))
        e = raw.set_index("record_id")
        mt = d[d.dup_reason == "MATCHED_PRIMARY"].head(3)
        print("\n--- Example 3: matched pair -> same purchase? (tool net amount vs ERP amount)")
        for _, r in mt.iterrows():
            print(f"{r.record_id}: net {e.at[r.record_id,'spend_raw']:,.2f} | {r.matched_to}: {e.at[r.matched_to,'spend_raw']:,.2f} | text: {str(e.at[r.record_id,'description'])[:30]!r} vs {str(e.at[r.matched_to,'description'])[:30]!r}")
print("\n--- Example 4: classified lines")
print(inc.merge(raw[["record_id", "description"]], on="record_id").sample(5, random_state=1)[["description", "l1", "l2", "l3", "confidence"]].to_string(index=False))
print("\nOVERALL:", "ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED, see FAIL lines above")
