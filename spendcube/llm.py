"""Optional LLM classification. Works with several providers over plain HTTPS (no SDK needed):
  - gemini    : Google AI Studio free tier (env GEMINI_API_KEY)               <- free, no card
  - groq      : Groq free tier (env GROQ_API_KEY)
  - ollama    : local model on your own machine, no key, data never leaves it
  - anthropic : Claude API (env ANTHROPIC_API_KEY, paid)
Config classify.llm: {enabled: auto|true|false, provider: auto|gemini|groq|ollama|anthropic, model: ..., rpm: ...}
'auto' picks the first provider whose key is present. No key (or any failure) -> local model, so the tool always runs.
Answers are cached in .llm_cache.json, so re-runs cost no calls."""
import hashlib, json, os, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

CACHE_FILE = ".llm_cache.json"
PRESETS = {   # model names / limits change over time: override in config (classify.llm.model / rpm / base_url)
    "gemini": {"kind": "openai", "env": "GEMINI_API_KEY", "model": None, "rpm": 10, "workers": 1,   # model None = auto-discover
               "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"},
    "groq": {"kind": "openai", "env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile", "rpm": 20, "workers": 2,
             "url": "https://api.groq.com/openai/v1/chat/completions"},
    "ollama": {"kind": "openai", "env": None, "model": "llama3.1:8b", "rpm": 600, "workers": 1,
               "url": "http://localhost:11434/v1/chat/completions"},
    "anthropic": {"kind": "anthropic", "env": "ANTHROPIC_API_KEY", "model": "claude-haiku-5-5", "rpm": 300, "workers": 6,
                  "url": "https://api.anthropic.com/v1/messages"},
}
AUTO_ORDER = ["anthropic", "gemini", "groq"]          # ollama only when asked for explicitly


def resolve(lcfg):
    """-> provider settings dict or None."""
    mode = lcfg.get("enabled", "auto")
    if mode is False or str(mode).lower() == "false":
        return None
    name = lcfg.get("provider", "auto")
    if name == "auto":
        name = next((p for p in AUTO_ORDER if os.environ.get(PRESETS[p]["env"])), None)
        if name is None:
            return None
    if name not in PRESETS:
        return None
    s = dict(PRESETS[name], name=name)
    for k in ("model", "rpm", "workers", "base_url"):
        if lcfg.get(k) is not None:
            s["url" if k == "base_url" else k] = lcfg[k]
    s["key"] = os.environ.get(lcfg.get("api_key_env") or s["env"] or "", "") if s["env"] or lcfg.get("api_key_env") else ""
    if s["env"] and not s["key"]:
        return None
    return s


def llm_available(lcfg):
    return resolve(lcfg) is not None


class _Pace:
    """Keeps us under the provider's requests-per-minute limit (free tiers are strict)."""
    def __init__(self, rpm):
        self.gap, self.t, self.lock = 60.0 / max(rpm, 1), 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            d = self.t + self.gap - time.time()
            if d > 0:
                time.sleep(d)
            self.t = time.time()


def _call(s, pace, prompt, retries=5, quiet=False):
    for attempt in range(retries):
        pace.wait()
        try:
            if s["kind"] == "anthropic":
                body = {"model": s["model"], "max_tokens": 2000, "messages": [{"role": "user", "content": prompt}]}
                hdr = {"x-api-key": s["key"], "anthropic-version": "2023-06-01", "content-type": "application/json"}
            else:
                body = {"model": s["model"], "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
                hdr = {"content-type": "application/json"}
                if s["key"]:
                    hdr["authorization"] = "Bearer " + s["key"]
            req = urllib.request.Request(s["url"], data=json.dumps(body).encode(), headers=hdr)
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            txt = resp["content"][0]["text"] if s["kind"] == "anthropic" else resp["choices"][0]["message"]["content"]
            return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
        except Exception as e:
            code = getattr(e, "code", None)
            if code in (401, 403, 404):                       # wrong key / model name: retrying will not help
                if not quiet:
                    print(f"[llm] {s['name']} rejected the request (HTTP {code}) for model {s['model']}: check the key and model name")
                return None
            if attempt == retries - 1:
                print(f"[llm] batch failed after {retries} tries: {type(e).__name__} {code or ''}")
                return None
            time.sleep(15 * (attempt + 1) if code == 429 else 3 * (attempt + 1))   # 429 = rate limit: wait longer


def _gemini_candidates(key):
    """Ask Google which models this key can use; return text models, best guess first (stable lite, stable flash, previews)."""
    import re
    url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200&key=" + key
    with urllib.request.urlopen(url, timeout=60) as r:
        models = json.loads(r.read()).get("models", [])
    bad = re.compile(r"image|tts|live|audio|embed|transcribe|omni|robotics|veo|imagen|native|computer|thinking|exp|learnlm|gemma|aqa")
    cands = []
    for m in models:
        name = m.get("name", "").replace("models/", "")
        mm = re.match(r"gemini-(\d+(?:\.\d+)?)-flash(-lite)?(-preview.*)?$", name)
        if not mm or bad.search(name) or "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        cands.append(((1 if mm.group(3) else 0, 0 if mm.group(2) else 1, -float(mm.group(1))), name))
    return [n for _, n in sorted(cands)]


def _pick_gemini_model(s, pace):
    try:
        cands = _gemini_candidates(s["key"])
    except Exception as e:
        print(f"[llm] could not list Gemini models ({type(e).__name__} {getattr(e, 'code', '')}): check the key")
        return None
    print(f"[llm] models this key can see: {cands[:6]}")
    for name in cands[:6]:
        s["model"] = name
        if _call(s, pace, 'Reply with only this JSON: {"ok": 1}', retries=2, quiet=True):
            return name
    return None


def llm_leaf_choices(texts, paths, lcfg):
    """texts: unique line descriptions. Returns {text_index: leaf_index}; partial results are fine."""
    s = resolve(lcfg)
    if s is None:
        return {}
    batch = lcfg.get("batch_size", 40)
    if s["model"] is None:
        s["model"] = _pick_gemini_model(s, _Pace(s["rpm"]))
        if s["model"] is None:
            print("[llm] no usable Gemini model found -> using the local classifier. Set classify.llm.model by hand if you know one.")
            return {}
    tree = "\n".join(f"{i}: " + " > ".join(p) for i, p in enumerate(paths))
    tree_id = hashlib.md5((tree + s["model"]).encode()).hexdigest()[:10]
    try:
        cache = json.load(open(CACHE_FILE))
    except Exception:
        cache = {}
    ck = lambda t: tree_id + "|" + hashlib.md5(t.encode()).hexdigest()
    out = {i: cache[ck(t)] for i, t in enumerate(texts) if ck(t) in cache}
    todo = [i for i in range(len(texts)) if i not in out]
    calls = -(-len(todo) // batch)
    print(f"[llm] provider={s['name']} model={s['model']} | {len(texts)} unique descriptions, {len(out)} cached, "
          f"{len(todo)} to ask in ~{calls} calls (~{calls * 60 / max(s['rpm'], 1):.0f}s at {s['rpm']} requests/min)")
    pace = _Pace(s["rpm"])
    chunks = [todo[k:k + batch] for k in range(0, len(todo), batch)]

    def work(idx):
        items = "\n".join(f"{j}: {texts[i][:160]}" for j, i in enumerate(idx))
        prompt = (f"You classify procurement/spend line descriptions into a category tree.\n"
                  f"Category tree (id: path from top level to lowest level):\n{tree}\n\n"
                  f"Descriptions (id: text). They may be abbreviated, misspelt, or in other languages:\n{items}\n\n"
                  "For each description choose the single best category id (always pick one, even if unsure). "
                  'Reply with ONLY a JSON object {"<description id>": <category id>, ...}.')
        return idx, _call(s, pace, prompt)

    with ThreadPoolExecutor(max_workers=s["workers"]) as ex:
        for idx, obj in ex.map(work, chunks):
            if not obj:
                continue
            for j, k in obj.items():
                try:
                    j, k = int(j), int(k)
                    if 0 <= j < len(idx) and 0 <= k < len(paths):
                        out[idx[j]] = k
                        cache[ck(texts[idx[j]])] = k
                except (ValueError, TypeError):
                    pass
            try:                                              # save progress after every batch (safe to stop and re-run)
                json.dump(cache, open(CACHE_FILE, "w"))
            except Exception:
                pass
    print(f"[llm] classified {len(out)} of {len(texts)} descriptions with the LLM")
    return out