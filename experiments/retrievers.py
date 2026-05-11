"""Retrievers for the three RAG variants (Variants 2 / 3 / 4).

In-memory KG implementation:
  - stage2 (1.68M triples) → adjacency lists for fast 1-hop / 2-hop queries.
  - stage3 (same triples, with conditions) → same adjacency + edge-with-conditions
    inverted list, replicating mediq_graphrag_test.py CYPHER_31 / 32 / 33.

UMLS matching reuses the project's EntityMatcher with a persistent disk cache
keyed by (surface_form, normalized_form, semantic_group) so repeated entities
across MediQ samples cost only one API call per unique term.
"""

import json
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from entity_matcher import EntityMatcher
from semantic_types import load_semantic_groups_from_file
from UMLS_KG.umls_client import UMLSClient

from experiments.config import ExperimentConfig
from experiments.llm_backend import OpenAIEmbedder, cosine_top_k

logger = logging.getLogger(__name__)

_CUI_RE = re.compile(r"^C\d{7}$")


# ──────────────────────────────────────────────────────────────────────
# Retrieval result envelope
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    formatted_text: str               # the block the runner pastes into the prompt
    n_items: int = 0                  # how many retrieved items
    strategy: str = ""                # variant-specific tag
    debug: dict = field(default_factory=dict)


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    def retrieve(self, *args, **kwargs) -> RetrievalResult: ...


# ──────────────────────────────────────────────────────────────────────
# Variant 2 — Vector RAG over Stage-0 recommendations
# ──────────────────────────────────────────────────────────────────────

class VectorRAGRetriever(BaseRetriever):
    name = "vector_rag"

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        recs_data = _load_json(cfg.stage0_recs_path)
        self.recommendations: list[dict] = recs_data.get("recommendations", [])
        # Stage 0 may store the sentence text under several keys depending on the
        # source DB (CREST uses `text`/`guideline_context`; PUBMED uses `abstract`;
        # the normalizer adds `raw_text`). Fall back across them so VectorRAG works
        # regardless of which DB produced the records.
        self.texts: list[str] = [
            (r.get("text")
             or r.get("raw_text")
             or r.get("abstract")
             or r.get("guideline_context")
             or "")
            for r in self.recommendations
        ]
        logger.info(f"VectorRAG: loaded {len(self.texts)} recommendations")

        self.embedder = OpenAIEmbedder(
            api_key=cfg.openai_api_key,
            model=cfg.embedding_model,
            batch_size=cfg.embedding_batch_size,
        )
        self.rec_matrix = self.embedder.embed_with_cache(
            self.texts, cfg.embedding_cache_path
        )
        # Per-query cache: text-sha256 → vector
        self._qcache_path = cfg.query_embedding_cache_path
        self._qcache: dict = {}
        self._qcache_lock = threading.Lock()
        self._load_qcache()

    def _load_qcache(self):
        if os.path.isfile(self._qcache_path):
            try:
                z = np.load(self._qcache_path, allow_pickle=False)
                keys = list(z["keys"])
                mat = z["matrix"]
                for i, k in enumerate(keys):
                    self._qcache[str(k)] = mat[i]
                logger.info(f"Loaded {len(self._qcache)} cached query embeddings")
            except Exception as e:
                logger.warning(f"Query embedding cache read failed ({e})")

    def _save_qcache(self):
        if not self._qcache:
            return
        try:
            keys = np.array(list(self._qcache.keys()))
            mat = np.stack(list(self._qcache.values()), axis=0)
            os.makedirs(os.path.dirname(self._qcache_path) or ".", exist_ok=True)
            np.savez(self._qcache_path, keys=keys, matrix=mat)
        except Exception as e:
            logger.warning(f"Query embedding cache save failed: {e}")

    def _embed_query(self, query: str) -> np.ndarray:
        import hashlib
        key = hashlib.sha256(query.encode("utf-8")).hexdigest()
        with self._qcache_lock:
            cached = self._qcache.get(key)
        if cached is not None:
            return cached
        vec = self.embedder.embed_texts([query])[0]
        with self._qcache_lock:
            self._qcache[key] = vec
        return vec

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        top_k = top_k or self.cfg.vector_top_k
        qvec = self._embed_query(query)
        hits = cosine_top_k(qvec, self.rec_matrix, top_k=top_k)

        lines = [f"# Retrieval strategy: vector_top{top_k} (text-embedding-3-large)"]
        retrieved = []
        for rank, (idx, sim) in enumerate(hits, 1):
            rec = self.recommendations[idx]
            text = rec.get("text", "")
            gid = rec.get("guideline_id", "")
            strength = rec.get("strength", "")
            lines.append(
                f"{rank}. (sim={sim:.3f}) {text}  "
                f"| guideline: {gid}  | strength: {strength}"
            )
            retrieved.append({
                "rank": rank,
                "similarity": sim,
                "recommendation_index": idx,
                "text": text,
                "guideline_id": gid,
                "strength": strength,
            })
        if len(retrieved) == 0:
            lines.append("(no recommendations retrieved)")
        return RetrievalResult(
            formatted_text="\n".join(lines),
            n_items=len(retrieved),
            strategy="vector_top_k",
            debug={"hits": retrieved},
        )

    def flush_cache(self):
        self._save_qcache()


# ──────────────────────────────────────────────────────────────────────
# Shared KG storage
# ──────────────────────────────────────────────────────────────────────

class KGStore:
    """Adjacency lists over a (head, relation, tail) triple list.

    head_id and tail_id are stored as-is from Stage 2/3 JSON; only head_id is
    guaranteed to be a CUI. Both endpoint ids are indexed for incidence queries.
    """

    def __init__(self, triples: list[dict]):
        self.triples: list[dict] = triples
        # Incidence: node_id → list of triple indices touching it.
        self.adj: dict[str, list[int]] = {}
        # Triples that carry at least one condition (for variant 4's 3-3).
        self.cond_triple_indices: list[int] = []
        for i, t in enumerate(triples):
            h = (t.get("head_cui") or "").strip()
            tl = (t.get("tail_id") or "").strip()
            if h:
                self.adj.setdefault(h, []).append(i)
            if tl and tl != h:
                self.adj.setdefault(tl, []).append(i)
            if t.get("has_conditions"):
                self.cond_triple_indices.append(i)

    def cuis_present(self, cuis: list[str]) -> list[str]:
        return [c for c in cuis if c in self.adj]

    def incident_edges(self, node_id: str) -> list[int]:
        return self.adj.get(node_id, [])


def _load_json(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────
# Variant 3 — KG (no conditions)
# ──────────────────────────────────────────────────────────────────────

class KGNoConditionsRetriever(BaseRetriever):
    name = "kg_no_cond"

    def __init__(self, cfg: ExperimentConfig, store: Optional[KGStore] = None):
        self.cfg = cfg
        if store is None:
            data = _load_json(cfg.stage2_triples_path)
            self.store = KGStore(data.get("triples", []))
        else:
            self.store = store
        logger.info(
            f"KGNoCond: {len(self.store.triples)} triples, "
            f"{len(self.store.adj)} unique node ids indexed"
        )

    def retrieve(self, cuis: list[str]) -> RetrievalResult:
        present = self.store.cuis_present(cuis)
        debug = {"queried_cuis": cuis, "present_cuis": present}
        if not present:
            return RetrievalResult(
                formatted_text="(no matched CUIs in KG)",
                n_items=0,
                strategy="empty",
                debug=debug,
            )

        # ── 1-hop edges incident to any matched CUI ──
        one_hop_idx: list[int] = []
        seen = set()
        for c in present:
            for idx in self.store.incident_edges(c):
                if idx not in seen:
                    seen.add(idx)
                    one_hop_idx.append(idx)
                    if len(one_hop_idx) >= self.cfg.kg_no_cond_one_hop_limit:
                        break
            if len(one_hop_idx) >= self.cfg.kg_no_cond_one_hop_limit:
                break

        # ── 2-hop paths between matched-CUI pairs ──
        paths: list[dict] = []
        if self.cfg.kg_no_cond_paths_max_hops >= 2 and len(present) >= 2:
            present_set = set(present)
            for src in present:
                if len(paths) >= self.cfg.kg_no_cond_paths_limit:
                    break
                # one step out from src
                for e1 in self.store.incident_edges(src):
                    t1 = self.store.triples[e1]
                    mid = _other_endpoint(t1, src)
                    if not mid or mid in present_set:
                        continue
                    # second step from mid → any other matched CUI
                    for e2 in self.store.incident_edges(mid):
                        if e2 == e1:
                            continue
                        t2 = self.store.triples[e2]
                        dst = _other_endpoint(t2, mid)
                        if dst and dst != src and dst in present_set:
                            paths.append({"e1": e1, "e2": e2, "mid": mid,
                                          "src": src, "dst": dst})
                            if len(paths) >= self.cfg.kg_no_cond_paths_limit:
                                break
                    if len(paths) >= self.cfg.kg_no_cond_paths_limit:
                        break

        # ── Format ──
        lines = [f"# Retrieval strategy: kg_no_conditions (1-hop + 2-hop paths)"]
        lines.append(f"# Matched CUIs: {len(present)} → {present}")
        lines.append("")
        lines.append(f"[1-HOP EDGES — {len(one_hop_idx)}]")
        for i, idx in enumerate(one_hop_idx, 1):
            lines.append("  " + _fmt_triple(i, self.store.triples[idx]))
        lines.append("")
        lines.append(f"[2-HOP PATHS — {len(paths)}]")
        for i, p in enumerate(paths, 1):
            t1 = self.store.triples[p["e1"]]
            t2 = self.store.triples[p["e2"]]
            lines.append(
                f"  {i}. ({_endpoint_name(t1, p['src'])}) "
                f"-[{t1.get('relation','')}]-> ({_endpoint_name(t1, p['mid'])}) "
                f"-[{t2.get('relation','')}]-> ({_endpoint_name(t2, p['dst'])})"
            )

        debug.update({
            "one_hop_triple_indices": one_hop_idx,
            "n_one_hop": len(one_hop_idx),
            "n_paths": len(paths),
        })
        return RetrievalResult(
            formatted_text="\n".join(lines),
            n_items=len(one_hop_idx) + len(paths),
            strategy="kg_no_cond_1hop+2hop",
            debug=debug,
        )


# ──────────────────────────────────────────────────────────────────────
# Variant 4 — KG with conditions (3-1 → 3-2 → 3-3 cascade)
# ──────────────────────────────────────────────────────────────────────

class KGWithConditionsRetriever(BaseRetriever):
    name = "kg_with_cond"

    def __init__(self, cfg: ExperimentConfig, store: Optional[KGStore] = None):
        self.cfg = cfg
        if store is None:
            data = _load_json(cfg.stage3_triples_path)
            self.store = KGStore(data.get("triples", []))
        else:
            self.store = store
        logger.info(
            f"KGWithCond: {len(self.store.triples)} triples, "
            f"{len(self.store.cond_triple_indices)} with conditions"
        )

    def retrieve(
        self,
        cuis: list[str],
        cond_keywords: list[str],
    ) -> RetrievalResult:
        debug = {
            "queried_cuis": cuis,
            "cond_keywords": cond_keywords,
            "attempts": [],
        }
        present = self.store.cuis_present(cuis)
        debug["present_cuis"] = present

        # ── 3-1: present CUIs AND condition keyword hit ──
        if present and cond_keywords:
            hits = self._scan(
                node_ids=present,
                keyword_filter=cond_keywords,
                require_conditions=True,
                limit=self.cfg.kg_with_cond_31_limit,
            )
            debug["attempts"].append({"strategy": "3-1", "rows": len(hits)})
            if hits:
                return _format_kg_rows(
                    hits, self.store, "3-1_entity+condition", debug
                )

        # ── 3-2: present CUIs only (no condition filter) ──
        if present:
            hits = self._scan(
                node_ids=present,
                keyword_filter=None,
                require_conditions=False,
                limit=self.cfg.kg_with_cond_32_limit,
            )
            debug["attempts"].append({"strategy": "3-2", "rows": len(hits)})
            if hits:
                return _format_kg_rows(
                    hits, self.store, "3-2_entity_1hop", debug
                )

        # ── 3-3: no CUI match — surface condition matches globally ──
        if cond_keywords:
            hits = self._scan_conditions(
                cond_keywords, limit=self.cfg.kg_with_cond_33_limit
            )
            debug["attempts"].append({"strategy": "3-3", "rows": len(hits)})
            if hits:
                return _format_kg_rows(
                    hits, self.store, "3-3_similar_conditions", debug
                )

        return RetrievalResult(
            formatted_text="(no relevant knowledge graph triples retrieved)",
            n_items=0,
            strategy="empty",
            debug=debug,
        )

    def _scan(
        self,
        node_ids: list[str],
        keyword_filter: Optional[list[str]],
        require_conditions: bool,
        limit: int,
    ) -> list[int]:
        """Iterate edges incident to node_ids; optionally filter by keyword."""
        hits: list[int] = []
        seen: set[int] = set()
        for node in node_ids:
            for idx in self.store.incident_edges(node):
                if idx in seen:
                    continue
                seen.add(idx)
                t = self.store.triples[idx]
                if require_conditions and not t.get("has_conditions"):
                    continue
                if keyword_filter and not _matches_any_keyword(t, keyword_filter):
                    continue
                hits.append(idx)
                if len(hits) >= limit:
                    return hits
        return hits

    def _scan_conditions(self, keywords: list[str], limit: int) -> list[int]:
        """Variant 4 / 3-3: scan condition-bearing edges by keyword.

        Mirrors CYPHER_33 in mediq_graphrag_test.py: returns at most one
        sample triple per unique conditions_json string, so the model sees
        diverse conditions rather than many edges sharing the same condition.
        """
        seen_conds: set[str] = set()
        hits: list[int] = []
        for idx in self.store.cond_triple_indices:
            t = self.store.triples[idx]
            if not _matches_any_keyword(t, keywords):
                continue
            key = t.get("conditions_json", "") or ""
            if key in seen_conds:
                continue
            seen_conds.add(key)
            hits.append(idx)
            if len(hits) >= limit:
                break
        return hits


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _other_endpoint(triple: dict, node_id: str) -> str:
    h = (triple.get("head_cui") or "").strip()
    t = (triple.get("tail_id") or "").strip()
    if h == node_id:
        return t
    if t == node_id:
        return h
    return ""


def _endpoint_name(triple: dict, node_id: str) -> str:
    if (triple.get("head_cui") or "").strip() == node_id:
        return triple.get("head_name", "") or node_id
    if (triple.get("tail_id") or "").strip() == node_id:
        return triple.get("tail_name", "") or node_id
    return node_id


def _matches_any_keyword(triple: dict, keywords: list[str]) -> bool:
    cj = (triple.get("conditions_json") or "").lower()
    if not cj or cj == "[]":
        return False
    return any(kw in cj for kw in keywords)


def _fmt_triple(rank: int, t: dict) -> str:
    parts = [
        f"{rank}. ({t.get('head_name','')}) "
        f"-[{t.get('relation','')}]-> ({t.get('tail_name','')})"
    ]
    gid = (t.get("condition_source") or {}).get("guideline_id", "") or ""
    if gid:
        parts.append(f"  | guideline: {gid}")
    strength = t.get("recommendation_strength")
    if strength:
        parts.append(f"  | strength: {strength}")
    cj = t.get("conditions_json", "[]")
    if cj and cj != "[]":
        parts.append(f"  | conditions: {cj}")
    return "".join(parts)


def _format_kg_rows(
    indices: list[int],
    store: KGStore,
    strategy: str,
    debug: dict,
) -> RetrievalResult:
    lines = [f"# Retrieval strategy: {strategy}"]
    for i, idx in enumerate(indices, 1):
        lines.append(_fmt_triple(i, store.triples[idx]))
    debug["selected_triple_indices"] = indices
    return RetrievalResult(
        formatted_text="\n".join(lines),
        n_items=len(indices),
        strategy=strategy,
        debug=debug,
    )


# ──────────────────────────────────────────────────────────────────────
# Cached UMLS matcher
# ──────────────────────────────────────────────────────────────────────

class CachedUMLSMatcher:
    """EntityMatcher + persistent disk cache (entity-form-keyed)."""

    def __init__(self, cfg: ExperimentConfig):
        if not cfg.umls_api_key:
            raise ValueError(
                "UMLS_API_KEY is required for entity → CUI matching (Variants 3, 4)."
            )
        # The UMLSClient reads UMLS_API_KEY from project_config; mirror it.
        import config as project_config
        project_config.UMLS_API_KEY = cfg.umls_api_key

        tui_to_group, _ = load_semantic_groups_from_file(cfg.semantic_groups_file)
        self.tui_to_group = tui_to_group
        self.client = UMLSClient()
        self.matcher = EntityMatcher(self.client, tui_to_group)
        self.cache_path = cfg.umls_match_cache_path
        self._cache: dict = {}
        self._lock = threading.Lock()
        self._dirty = False
        if os.path.isfile(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(
                    f"Loaded {len(self._cache)} cached UMLS match entries"
                )
            except Exception as e:
                logger.warning(f"UMLS cache read failed ({e})")

    @staticmethod
    def _key(entity: dict) -> str:
        return "||".join([
            (entity.get("surface_form") or "").strip().lower(),
            (entity.get("normalized_form") or "").strip().lower(),
            (entity.get("semantic_group") or "").strip().upper(),
        ])

    def match(self, entity: dict) -> dict:
        key = self._key(entity)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            result = self.matcher.match_entity(entity)
        except Exception as e:
            logger.warning(
                f"UMLS match failed for {entity.get('surface_form')!r}: {e}"
            )
            return {"entity": entity, "matched": False, "matches": [], "match_type": "error"}
        with self._lock:
            self._cache[key] = result
            self._dirty = True
        return result

    def match_many(self, entities: list[dict]) -> tuple[list[dict], list[str]]:
        """Returns (per-entity match results, deduped CUI list)."""
        results: list[dict] = []
        cuis: list[str] = []
        seen_cuis: set[str] = set()
        for ent in entities:
            r = self.match(ent)
            results.append(r)
            if r.get("matched"):
                for m in r.get("matches", []):
                    cui = m.get("cui", "")
                    if _CUI_RE.match(cui) and cui not in seen_cuis:
                        seen_cuis.add(cui)
                        cuis.append(cui)
        return results, cuis

    def flush(self):
        if not self._dirty:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, default=str)
            self._dirty = False
        except Exception as e:
            logger.warning(f"UMLS cache save failed: {e}")
