# APPROACH

**Pipeline.** config -> ingest every source into one unified frame (raw columns kept) -> clean (parse numbers,
strip VAT where gross, convert to reporting currency, parse dates with per-config formats + flags) -> match systems
(count once) -> exclusion rules + reason codes -> supplier_key / item_key -> classify included lines -> CSV.

**Matching (A).** (1) A tool line repeated inside the tool (same invoice digits, PO, net amount, text, qty) is
`DUPLICATE`. (2) PO is extracted by regex from all five reformatted shapes; a tool line is matched one-to-one to the
unused ERP line on the same PO whose net amount agrees within 0.1%. (3) Second pass on the same PO: same quantity and
near-identical text but different amount (price variance). (4) Lines with no PO / unknown PO: same net amount (+-0.1)
and quantity and text similarity >= 0.6. Matched tool lines are `DUPLICATE`; unmatched ones are tool-only spend and
are counted. Matched pairs have mean text similarity 0.98.

**Supplier de-dup.** Names are normalised (case, punctuation, legal suffixes), then clustered space-insensitively with
a similarity threshold (handles "AL TON"/"ALTON"); vendor ids link clusters. Names with no vendor-master id form one
`unmapped` bucket (a convention visible in A's labels, see DECISIONS.md). A, labels hidden: pairwise F1 0.981.
**Item de-dup.** Same item code = same item; placeholder codes (catalogue / free-text lines) share one bucket (config);
code-less lines join the nearest coded item by char-n-gram TF-IDF or form text clusters. A: pairwise F1 0.995.

**Classifying into an unseen tree.** The tree is read from file. Zero-label mode: each leaf becomes a document made
of its path names (+ acronyms such as "Continuous Miners" -> "cm"); lines are scored by word+char TF-IDF similarity;
the most confident 40% become pseudo-labels, a logistic regression is trained on them and blended with the
similarity prior (2 rounds); the leaf is chosen hierarchically (best L1 by summed probability, then best leaf in it).
If a labels file exists the same features train a supervised model. Optional LLM and sentence-embedding hooks exist.

**Measured accuracy (A, labels hidden, tree only):** L1 55% / L2 20% / L3 16% unweighted; spend-weighted 72% / 29% / 24%.
B zero-label (seed hidden): L1 90% / L2 87% / L3 30%. **This is the weak spot**: without labels or an LLM/embedding
model the local classifier only reaches tree-vocabulary level. Supervised hold-out (A, new lines): L1 96% L3 90%;
(A, unseen descriptions): L1 89% L3 77%. B supervised hold-out L1 96% L3 93%.

**Runtime.** A 65k records ~70 s, B 41k ~30 s (CPU only), far inside 60 min / 25k lines.
**LLM.** Optional: used automatically only when ANTHROPIC_API_KEY is set (unlabelled classification only), otherwise the local model runs. Tested with a mocked API for parsing, parallel batches, caching and fallback; not yet run against the live API, so no measured cost;
by design it sends only unique descriptions, ~40 per request, so expect cents per 1,000 lines on a small model.
Built with AI assistance (Claude) for code and analysis.

Live test (Gemini free tier, auto-selected gemini-3.5-flash-lite, 300 labelled lines, labels hidden): B local L1/L2/L3 = 86/84/28%, with LLM = 95/92/85%; A local = 60/26/20%, with LLM = 68/53/45%. Cost on free tier: 0. LLM is not used on A2/B2 (labels exist, supervised model is used).