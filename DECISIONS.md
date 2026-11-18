# DECISIONS (what the data showed, what I decided)

## Company A
1. **Net amounts.** ERP `value_zar` is net. Tool amounts are gross: `net = gross / (1 + VAT)`. VAT appears as `15`, `0.15`,
   `15%`, blank (6,542 rows) and `0` (1,580). Blank -> 15% (brief); an explicit 0 stays 0. Amounts like `5,266.63` parsed.
2. **PO shapes.** `4510…`, `PO#…`, `PO …`, `…/10.0`, `…/100.0`. 10-digit regex; item suffix used only to break ties.
3. **Count once.** ERP is counted; matched tool lines are `DUPLICATE`; a tool line exported twice is counted once.
   Tool-only lines (about 6k, ~381m ZAR) are included as spend that exists in one system only.
4. **Status.** Paid/Approved count. Cancelled -> `CANCELLED`; On Hold/Rejected -> `NOT_APPROVED`. Status wins over duplicate.
5. **Negative ERP lines (434)** are reversals/credit notes: kept, so total spend is net of reversals.
6. **No NON_PROCUREMENT in A.** I scanned both files for tax/intercompany/royalty/bank/fee keywords and the labels contain
   a category for every ERP line; hits were false friends ("VALVE BANK"). Rule list is empty (config-ready).
7. **Dates.** 65% of tool dates blank and 24k ERP rows have no posting date: flagged `missing`, never used for matching.
8. **ERP layouts.** One ERP file mixes two layouts (24,370 rows without vendor id/date/site). Handled by treating each field as optional.
9. **Supplier labels.** The golden key puts all suppliers without a vendor-master id (85% of those rows, 122 names) into one
   group. I reproduce this with `unmapped_bucket` (config option). Similarly golden `item_key` is the material number, except
   Catalogue/Free-Text lines whose placeholder codes share one key (config `bucket_rules`). Both are conventions learned from
   labels; for a new company without them the options are simply left off.

## Company B
1. **Currency.** `usd = amount / currency_rate`. USD lines are left alone even when a stray rate is present (1,279 rows).
   Missing rate (3,343 non-USD rows) -> `fx_rates_usd.csv` by posting month (nearest month if the currency has gaps).
2. **Dates.** `invoice_date` has 2018 and 2027 values (203 outside range, flagged). FX month uses `posting_date`, always valid.
3. **No status column** -> `CANCELLED`/`NOT_APPROVED` cannot occur from this file; config supports them if present.
4. **NON_PROCUREMENT** (learned from the seed, rule-based): intercompany GL descriptions (`IC06`, `I/C Payable`, `IC Payables`, ...)
   and tax lines (State & Local Taxes, Sales Tax, GST Input Tax). GL alone is not enough: `PEUS` holds mostly real spend.
   Rules reproduce all 450 seed exclusions (they were fitted on the seed, so this is not a held-out figure).
5. **Exact repeat rows (613)** are NOT excluded: only 2 seed rows are in repeat groups and the brief's count-once rule is about A.
6. **Blank descriptions (17%)** fall back to GL description / material number as classifier text.
7. **Missing amount (3 rows, currency RCN)**: kept as included with spend 0 and flagged.
8. Supplier key = vendor id (no names); not scored for B2.

## Known weaknesses
Zero-label classification is weak on domain jargon (see APPROACH.md). Supplier/item "buckets" are learned conventions.
