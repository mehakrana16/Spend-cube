"""Step 7: classify each included line into the company's tree (read from file at run time).
 - zero-label mode : taxonomy-text similarity -> confident pseudo-labels -> self-trained linear model
 - supervised mode : if a labels file is configured, train on it (same features)
Optional: local sentence-embedding model (config classify.embedding_model) adds a semantic similarity signal."""
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


def load_taxonomy(path, level_columns):
    t = pd.read_csv(path, dtype=str).fillna("")
    cols = [c for c in level_columns if c in t.columns]
    paths = []
    for _, r in t.iterrows():
        p = tuple(r[c].strip() for c in cols if r[c].strip())
        if p and p not in paths:
            paths.append(p)
    return paths, len(cols)


def _clean(s):
    """lower-case alnum tokens with a light plural stem (valves -> valve, batteries -> battery)."""
    if pd.isna(s):
        return ""
    t = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip()
    return re.sub(r"ies\b", "y", re.sub(r"(?<=[a-z]{3})s\b", "", t))


def _initials(path):
    """Acronyms of each tree name ('Continuous Miners' -> 'cm'): descriptions often abbreviate."""
    out = []
    for n in path:
        w = [x for x in _clean(n).split() if x not in ("and", "or", "of", "the", "for")]
        if len(w) > 1:
            out.append("".join(x[0] for x in w))
    return out


def _text_frame(df, ccfg):
    main = df["description"].map(_clean)
    extra_cols = [c for c in (ccfg.get("extra_text") or []) if c in df.columns]
    extra = df[extra_cols].fillna("").astype(str).agg(" ".join, axis=1).map(_clean) if extra_cols else pd.Series("", index=df.index)
    main_f = main.where(main != "", extra)                       # blank description -> fall back to other fields
    return main_f, (main_f + " " + extra).str.strip()


def _features(fit_texts, texts_list):
    w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    w.fit(fit_texts); c.fit(fit_texts)
    return [normalize(hstack([w.transform(t), c.transform(t)]).tocsr()) for t in texts_list]


def _softmax(z, T):
    z = z * T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _embedding_sim(texts, leaf_docs, model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    m = SentenceTransformer(model_name)
    a = m.encode(list(texts), normalize_embeddings=True, batch_size=128)
    b = m.encode(list(leaf_docs), normalize_embeddings=True)
    return a @ b.T


def classify(df, paths, depth, ccfg, labels=None, seed=0):
    """df: included rows. labels: DataFrame(record_id + tuple 'path') or None.
    Returns DataFrame(index=df.index, columns l1..l{depth}, confidence, method)."""
    ccfg = ccfg or {}
    main, full = _text_frame(df, ccfg)
    K = len(paths)
    leaf_docs = [" ".join([_clean(n) for n in p for _ in (0, 1)] + _initials(p)) for p in paths]
    l1_of = np.array([p[0] for p in paths])
    uniq_full = full.unique()
    fit_texts = list(uniq_full) + leaf_docs
    (Xu, Xl) = _features(fit_texts, [uniq_full, leaf_docs])
    pos = pd.Series(np.arange(len(uniq_full)), index=uniq_full)

    method = "zero_label"
    sim = (Xu @ Xl.T).toarray()
    emb = _embedding_sim(uniq_full, leaf_docs, ccfg["embedding_model"]) if ccfg.get("embedding_model") else None
    if emb is not None:
        sim = 0.5 * sim + 0.5 * emb
        method = "zero_label+embeddings"

    rng = np.random.RandomState(seed)
    if labels is not None and len(labels):
        # --- supervised
        method = "supervised"
        lab = labels[labels["leaf"].isin(range(K))].copy()
        lab["full"] = full.reindex(lab.index).values
        lab = lab.dropna(subset=["full"])
        y, rows = lab["leaf"].values, pos.reindex(lab["full"]).values
        sample_w = None
        clf = LogisticRegression(C=20, max_iter=300, class_weight="balanced")
        classes_seen = np.unique(y)
        clf.fit(Xu[rows], y)
        P = np.zeros((len(uniq_full), K))
        P[:, clf.classes_] = clf.predict_proba(Xu)
        prior = _softmax(sim, 12.0)
        P = 0.9 * P + 0.1 * prior
    else:
        # --- zero-label self-training
        P = _softmax(sim, 12.0)
        for rnd in range(2):
            top2 = np.sort(P, axis=1)[:, -2:]
            conf, margin = top2[:, 1], top2[:, 1] - top2[:, 0]
            has_sig = sim.max(axis=1) > 0.02
            thr = np.quantile(margin[has_sig], 0.6) if has_sig.any() else 1
            sel = has_sig & (margin >= thr)
            y = P.argmax(axis=1)[sel]
            if len(np.unique(y)) < 2:
                break
            clf = LogisticRegression(C=10, max_iter=200, class_weight="balanced")
            clf.fit(Xu[sel], y)
            Pm = np.zeros_like(P)
            Pm[:, clf.classes_] = clf.predict_proba(Xu)
            P = 0.6 * Pm + 0.4 * _softmax(sim, 12.0)

    lcfg = ccfg.get("llm", {}) or {}
    from .llm import llm_available, llm_leaf_choices
    # LLM helps most when there are no labels; with labels the supervised model is already strong
    if llm_available(lcfg) and (method != "supervised" or lcfg.get("use_with_labels", False)):
        choice = llm_leaf_choices(list(uniq_full), paths, lcfg)        # {text_index: leaf_index} (may be partial)
        if choice:
            w = lcfg.get("weight", 0.85)
            for i, k in choice.items():
                P[i] = (1 - w) * P[i] / max(P[i].sum(), 1e-9)
                P[i, k] += w
            method += "+llm"

    # hierarchical decision: pick L1 by summed probability, then the best leaf inside it
    l1_names = pd.unique(l1_of)
    l1_mass = np.stack([P[:, l1_of == n].sum(axis=1) for n in l1_names], axis=1)
    best_l1 = l1_mass.argmax(axis=1)
    leaf = np.array([np.where(l1_of == l1_names[b])[0][P[i, l1_of == l1_names[b]].argmax()] for i, b in enumerate(best_l1)])
    conf = P[np.arange(len(P)), leaf] / np.clip(P.sum(axis=1), 1e-9, None)
    conf = np.clip(0.2 + 0.8 * conf, 0, 1) if not method.startswith("supervised") else np.clip(conf, 0, 1)

    rows_idx = pos.reindex(full.values).values
    res = pd.DataFrame(index=df.index)
    for d in range(depth):
        res[f"l{d+1}"] = [paths[leaf[r]][d] if d < len(paths[leaf[r]]) else "" for r in rows_idx]
    res["confidence"] = conf[rows_idx].round(4)
    res["leaf"] = leaf[rows_idx]
    res["method"] = method
    return res
