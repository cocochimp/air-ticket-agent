"""工作三：向量 + BM25 双路召回 → RRF 融合 → 硬过滤 → 精排（动态截断）→ MMR → 注入。每层留痕。"""
import logging
import math
import os
import re
import threading
from collections import Counter

import jieba

from . import data
from . import settings as S

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
MODEL_DIR = S.ROOT / "models"
EMB_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "ms-marco-MultiBERT-L-12"
CODE = re.compile(r"[A-Z][A-Z0-9_]+|\d+%?")
jieba.setLogLevel(logging.WARNING)
# BM25 停用词：在几乎每条规则里都出现的虚词/泛词，不应让无关问题凭它们命中
STOPWORDS = frozenset("航班 申请 规则 票 怎么 可以 吗 的 了 有 能 我 要 什么 多少 一下 请问".split())
# 退改领域词：问题里至少出现一个，BM25 命中才被当作相关信号
DOMAIN_TERMS = ("改签", "改期", "退票", "退款", "退", "改", "手续费", "费", "延误", "取消", "误机", "差价", "补差",
                "变更", "推迟", "不坐", "不去")
STAGE_LABEL = {"dense": "向量召回", "sparse": "BM25 召回", "rrf": "RRF 融合", "filter": "硬过滤",
               "rerank": "精排", "mmr": "MMR 去冗余", "inject": "注入生成"}


def has_domain_term(text: str) -> bool:
    """问题里是否出现退改相关词；没有的话属于无关问题。"""
    return any(t in (text or "") for t in DOMAIN_TERMS)

# 规则文本里没有、但旅客常说的同义说法：检索时补上规则里的正式用语（只影响召回，不改返回的 query）
SYNONYMS = {"推迟": "改签 改期", "延期": "改签 改期", "改天": "改签 改期", "换一天": "改签 改期",
            "变更": "改签", "更改": "改签 改期", "退订": "退票", "不坐": "退票", "不去": "退票"}

_lock = threading.Lock()          # 保护 _index
_models_lock = threading.Lock()   # 保护模型加载，避免并发重复加载
_models = {"loaded": False, "embed": None, "rerank": None, "error": ""}
_index = {"key": None, "value": None}


def tokenize(text: str) -> list[str]:
    words = [w for w in jieba.lcut(text)
             if w.strip() and w not in STOPWORDS and not re.fullmatch(r"[\W_]+", w)]
    return words + CODE.findall(text)


def _load_models():
    if _models["loaded"]:
        return
    with _models_lock:
        if _models["loaded"]:       # 双重检查：等锁期间别的线程可能已加载完
            return
        try:
            from fastembed import TextEmbedding
            from flashrank import Ranker
            embed = TextEmbedding(EMB_MODEL, cache_dir=str(MODEL_DIR))
            rerank = Ranker(model_name=RERANK_MODEL, cache_dir=str(MODEL_DIR))
            _models.update(embed=embed, rerank=rerank, error="")
        except Exception as e:  # 下载失败、离线等 → 降级
            _models.update(embed=None, rerank=None, error=f"{type(e).__name__}: {e}")
        _models["loaded"] = True    # 成功或失败都已记录后才置位


def _mark_failed(e: Exception):
    """查询期模型报错：标记失败，之后的检索降级到 fallback 并带上原因。"""
    with _models_lock:
        _models.update(loaded=True, embed=None, rerank=None, error=f"运行时失败 {type(e).__name__}: {e}")


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
        if not rules:               # 空规则库：所有召回层返回空
            self.bm25, self.vecs, self.idf = None, [], {}
            return
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
        if not self.rules:
            return []
        qv = self.query_vec(q)
        if self.mode == "hybrid":
            hits = self.qdrant.query_points("rules", query=qv, limit=k).points
            return [(h.id, float(h.score)) for h in hits]
        sims = [(i, _cos(qv, v)) for i, v in enumerate(self.vecs)]
        return sorted([x for x in sims if x[1] > 0], key=lambda x: -x[1])[:k]

    def sparse(self, q, k):
        if not self.rules:
            return []
        scores = self.bm25.get_scores(tokenize(q))
        return sorted([(i, float(s)) for i, s in enumerate(scores) if s > 0], key=lambda x: -x[1])[:k]


def expand_query(query: str) -> str:
    extra = [v for k, v in SYNONYMS.items() if k in query]
    return f"{query} {' '.join(extra)}" if extra else query


def invalidate():
    with _lock:
        _index.update(key=None, value=None)


def get_index() -> Index:
    version = data.rules_version()      # 先读版本再读规则：中途有改动时下次会重建
    rules = data.list_rules()
    mode = status()["backend"]
    key = (mode, str(data.db_path()), version, len(rules))
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


def _rerank(idx, query, cands, filters):
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
        out.append((i, rel if fs is None else 0.7 * rel + 0.3 * fs))
    return sorted(out, key=lambda x: -x[1])


def _boost(idx, items, boost, cfg):
    """画像加权：只作用于已经过了 rerank_min 的候选，不能把低于门槛的结果抬上来。"""
    out = []
    for i, score in items:
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
    try:
        return _search(query, filters, boost)
    except Exception as e:
        if status()["backend"] != "hybrid":
            raise
        _mark_failed(e)             # 模型在查询期出错：本次起降级为 fallback
        invalidate()
        return _search(query, filters, boost)


def _search(query: str, filters: dict | None, boost: dict | None) -> dict:
    cfg = S.get()["retrieval"]
    filters, boost = filters or {}, boost or {}
    idx = get_index()
    funnel = []
    raw_query, query = query, expand_query(query)

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
    has_domain = any(t in raw_query for t in DOMAIN_TERMS)

    def relevant(i):    # 绝对相关性门槛：领域词 + (BM25 命中 或 向量过 dense_min)，或向量过更严的 dense_strict
        d = dense_score.get(i, 0)
        return (has_domain and (i in sparse_ids or d >= cfg["dense_min"][idx.mode])) \
            or d >= cfg["dense_strict"][idx.mode]

    kept = [(i, s) for i, s in fused if _match(idx.rules[i], filters) and relevant(i)]
    stage("filter", None, kept)
    ranked = _rerank(idx, query, kept, filters)
    top = _boost(idx, [x for x in ranked if x[1] >= cfg["rerank_min"]], boost, cfg)[:cfg["k_rerank"]]
    stage("rerank", cfg["k_rerank"], top)
    diverse = _mmr(idx, top, cfg["mmr_lambda"], cfg["k_mmr"])
    stage("mmr", cfg["k_mmr"], diverse)
    inject = diverse[:cfg["k_inject"]]
    stage("inject", cfg["k_inject"], inject)
    st = status()
    return {"query": raw_query, "filters": filters,
            "rules": [{**{k: idx.rules[i][k] for k in ("id", "airline", "cabin", "fare_product", "scenario",
                                                        "title", "content")}, "score": round(s, 4)}
                      for i, s in inject],
            "funnel": funnel, "backend": idx.mode, "degraded": st["degraded"], "degraded_reason": st["reason"]}
