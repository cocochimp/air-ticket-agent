"""工作三：向量 + BM25 双路召回 → RRF 融合 → 硬过滤 → 精排（动态截断）→ MMR → 注入。每层留痕。"""
import logging
import math
import os
import re
import threading
from collections import Counter
from . import data
from . import settings as S

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
MODEL_DIR = S.ROOT / "models"
EMB_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "ms-marco-MultiBERT-L-12"
CODE = re.compile(r"[A-Z][A-Z0-9_]+|\d+%?")
STAGE_LABEL = {"dense": "向量召回", "sparse": "BM25 召回", "rrf": "RRF 融合", "filter": "硬过滤",
               "rerank": "精排", "mmr": "MMR 去冗余", "inject": "注入生成"}

_lock = threading.Lock()
_models = {"loaded": False, "embed": None, "rerank": None, "error": ""}
_index = {"key": None, "value": None}


def tokenize(text: str) -> list[str]:
    import jieba
    jieba.setLogLevel(logging.WARNING)
    words = [w for w in jieba.lcut(text) if w.strip() and not re.fullmatch(r"[\W_]+", w)]
    return words + CODE.findall(text)


def _load_models():
    if _models["loaded"]:
        return
    _models["loaded"] = True
    try:
        from fastembed import TextEmbedding
        from flashrank import Ranker
        _models["embed"] = TextEmbedding(EMB_MODEL, cache_dir=str(MODEL_DIR))
        _models["rerank"] = Ranker(model_name=RERANK_MODEL, cache_dir=str(MODEL_DIR))
    except Exception as e:  # 下载失败、离线等 → 降级
        _models.update(embed=None, rerank=None, error=f"{type(e).__name__}: {e}")


def status() -> dict:
    want = S.get()["retrieval"]["backend"]
    mode = "fallback"
    if want == "hybrid":
        _load_models()
        if _models["embed"] is not None:
            mode = "hybrid"
    return {"configured": want, "backend": mode, "degraded": want == "hybrid" and mode != "hybrid",
            "reason": _models["error"] if want == "hybrid" and mode != "hybrid" else ""}


# ---------- 降级用的字符 n-gram TF-IDF ----------
def _ngrams(t):
    t = re.sub(r"\s+", "", t)
    return [t[i:i + n] for n in (1, 2) for i in range(len(t) - n + 1)]


def _weigh(cnt, idf):
    v = {g: c * idf[g] for g, c in cnt.items() if g in idf}
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {g: x / n for g, x in v.items()}


def _cos(a, b):
    if isinstance(a, dict):
        return sum(x * b.get(g, 0.0) for g, x in a.items())
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class Index:
    def __init__(self, rules: list[dict], mode: str):
        from rank_bm25 import BM25Okapi
        self.rules, self.mode = rules, mode
        self.texts = [f"{r['title']}。{r['content']}" for r in rules]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        if mode == "hybrid":
            from qdrant_client import QdrantClient, models as qm
            self.vecs = [[float(x) for x in v] for v in _models["embed"].embed(self.texts)]
            self.qdrant = QdrantClient(":memory:")
            self.qdrant.create_collection(
                "rules", vectors_config=qm.VectorParams(size=len(self.vecs[0]), distance=qm.Distance.COSINE),
                hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=100))
            self.qdrant.upsert("rules", points=[qm.PointStruct(id=i, vector=v, payload={"rule_id": r["id"]})
                                                for i, (v, r) in enumerate(zip(self.vecs, rules))])
        else:
            docs = [Counter(_ngrams(t)) for t in self.texts]
            df = Counter(g for d in docs for g in d)
            n = len(docs)
            self.idf = {g: math.log((n + 1) / (c + 1)) + 1 for g, c in df.items()}
            self.vecs = [_weigh(d, self.idf) for d in docs]

    def query_vec(self, q):
        if self.mode == "hybrid":
            return [float(x) for x in next(iter(_models["embed"].query_embed(q)))]
        return _weigh(Counter(_ngrams(q)), self.idf)

    def dense(self, q, k):
        qv = self.query_vec(q)
        if self.mode == "hybrid":
            hits = self.qdrant.query_points("rules", query=qv, limit=k).points
            return [(h.id, float(h.score)) for h in hits]
        sims = [(i, _cos(qv, v)) for i, v in enumerate(self.vecs)]
        return sorted([x for x in sims if x[1] > 0], key=lambda x: -x[1])[:k]

    def sparse(self, q, k):
        scores = self.bm25.get_scores(tokenize(q))
        return sorted([(i, float(s)) for i, s in enumerate(scores) if s > 0], key=lambda x: -x[1])[:k]


def invalidate():
    with _lock:
        _index.update(key=None, value=None)


def get_index() -> Index:
    rules = data.list_rules()
    mode = status()["backend"]
    key = (mode, len(rules), max((r["updated_at"] for r in rules), default=""), tuple(r["id"] for r in rules))
    with _lock:
        if _index["key"] != key:
            _index.update(key=key, value=Index(rules, mode))
        return _index["value"]


def _match(rule, filters):
    for f in ("airline", "cabin", "fare_product"):
        if filters.get(f) and rule[f] not in (filters[f], "*"):
            return False
    return True


def _field_score(rule, filters):
    got, total = 0.0, 0.0
    for f in ("airline", "fare_product"):
        if filters.get(f):
            total += 1
            got += 1.0 if rule[f] == filters[f] else 0.5
    if filters.get("scenario"):
        total += 2
        got += 2.0 if rule["scenario"] == filters["scenario"] else 0.0
    return got / total if total else None


def _rerank(idx, query, cands, filters, boost, cfg):
    if not cands:
        return []
    if idx.mode == "hybrid":
        from flashrank import RerankRequest
        res = _models["rerank"].rerank(RerankRequest(query=query, passages=[{"id": i, "text": idx.texts[i]}
                                                                            for i, _ in cands]))
        raw = {int(r["id"]): float(r["score"]) for r in res}
    else:
        bm = idx.bm25.get_scores(tokenize(query))
        qv = idx.query_vec(query)
        raw = {i: float(bm[i]) + 5 * _cos(qv, idx.vecs[i]) for i, _ in cands}
    lo, hi = min(raw.values()), max(raw.values())
    out = []
    for i, _ in cands:
        rel = (raw[i] - lo) / (hi - lo) if hi > lo else 1.0
        fs = _field_score(idx.rules[i], filters)
        score = rel if fs is None else 0.7 * rel + 0.3 * fs
        r = idx.rules[i]
        if r["airline"] in boost.get("airlines", []) or r["fare_product"] in boost.get("products", []):
            score *= 1 + cfg["profile_boost"]
        out.append((i, score))
    return sorted(out, key=lambda x: -x[1])


def _mmr(idx, cands, lam, k):
    chosen, pool = [], list(cands)
    while pool and len(chosen) < k:
        def gain(c):
            div = max((_cos(idx.vecs[c[0]], idx.vecs[s[0]]) for s in chosen), default=0.0)
            return lam * c[1] - (1 - lam) * div
        best = max(pool, key=gain)
        chosen.append(best)
        pool.remove(best)
    return chosen


def search(query: str, filters: dict | None = None, boost: dict | None = None) -> dict:
    cfg = S.get()["retrieval"]
    filters, boost = filters or {}, boost or {}
    idx = get_index()
    funnel = []

    def stage(name, k, items):
        funnel.append({"stage": name, "label": STAGE_LABEL[name], "k": k, "count": len(items),
                       "items": [{"id": idx.rules[i]["id"], "title": idx.rules[i]["title"], "score": round(s, 4)}
                                 for i, s in items]})

    dense = idx.dense(query, cfg["k_recall"])
    stage("dense", cfg["k_recall"], dense)
    sparse = idx.sparse(query, cfg["k_recall"])
    stage("sparse", cfg["k_recall"], sparse)
    rrf = Counter()
    for ranked in (dense, sparse):
        for rank, (i, _) in enumerate(ranked):
            rrf[i] += 1 / (cfg["rrf_k"] + rank + 1)
    fused = rrf.most_common(cfg["k_fuse"])
    stage("rrf", cfg["k_fuse"], fused)
    sparse_ids = {i for i, _ in sparse}
    dense_score = dict(dense)
    kept = [(i, s) for i, s in fused if _match(idx.rules[i], filters)
            and (i in sparse_ids or dense_score.get(i, 0) >= cfg["dense_min"][idx.mode])]   # 绝对相关性门槛
    stage("filter", None, kept)
    ranked = _rerank(idx, query, kept, filters, boost, cfg)
    top = [x for x in ranked if x[1] >= cfg["rerank_min"]][:cfg["k_rerank"]]
    stage("rerank", cfg["k_rerank"], top)
    diverse = _mmr(idx, top, cfg["mmr_lambda"], cfg["k_mmr"])
    stage("mmr", cfg["k_mmr"], diverse)
    inject = diverse[:cfg["k_inject"]]
    stage("inject", cfg["k_inject"], inject)
    st = status()
    return {"query": query, "filters": filters,
            "rules": [{**{k: idx.rules[i][k] for k in ("id", "airline", "cabin", "fare_product", "scenario",
                                                        "title", "content")}, "score": round(s, 4)}
                      for i, s in inject],
            "funnel": funnel, "backend": idx.mode, "degraded": st["degraded"], "degraded_reason": st["reason"]}
