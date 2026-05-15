"""Build per-sample UMLS sub-docs from extracted entity JSONL files.

Pipeline
1. Read `extract_info.py` outputs line-by-line.
2. Treat extracted entities as `main_entity`.
3. Match each main entity to UMLS concepts via `entity_matcher.py`.
4. Fetch 1-hop UMLS relation triples via `subgraph_builder.py`.
5. Write augmented JSONL with `sub_docs` per sample.

Design goals:
- low memory: streaming JSONL I/O, bounded batch size, no full-file loads
- parallelizable: sample-level thread pool with deterministic output order
- observable progress: structured logging for batch/file/global progress
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli_utils import setup_logging

import config as project_config
from entity_matcher import EntityMatcher
from experiments.config import ExperimentConfig
from semantic_types import load_semantic_groups_from_file
from subgraph_builder import build_1hop_subgraph
from UMLS_KG.umls_client import UMLSClient

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = os.path.join(".", "MediQ", "extracted")
DEFAULT_OUTPUT_DIR = os.path.join(".", "MediQ", "subdocs")
DEFAULT_INPUT_FILES = (
    "all_craft_md.jsonl",
    "all_dev_good.jsonl",
    "medqa_dev_convo.jsonl",
)
_JSON_DUMP_KWARGS = {"ensure_ascii": False, "separators": (",", ":")}


def _append_unique(items: list, value):
    if value and value not in items:
        items.append(value)


def _atomic_write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _build_cfg(args: argparse.Namespace, dataset_path: str, output_dir: str) -> ExperimentConfig:
    cfg = ExperimentConfig(
        dataset_name="mediq_subdocs",
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_samples=args.max_samples,
        start_index=args.start_index,
    )
    if args.log_level:
        cfg.log_level = args.log_level
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    return cfg


class FileBackedUMLSMatcher:
    """EntityMatcher wrapper with one-cache-file-per-entity for low memory use."""

    def __init__(self, cfg: ExperimentConfig, cache_dir: str):
        if not cfg.umls_api_key:
            raise ValueError("UMLS_API_KEY is required to build sub-docs.")

        project_config.UMLS_API_KEY = cfg.umls_api_key
        self.api_key = cfg.umls_api_key
        self.tui_to_group, _ = load_semantic_groups_from_file(cfg.semantic_groups_file)
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._gate = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._local = threading.local()
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "match_errors": 0,
        }

    def _matcher(self) -> EntityMatcher:
        matcher = getattr(self._local, "matcher", None)
        if matcher is None:
            client = UMLSClient(api_key=self.api_key)
            matcher = EntityMatcher(client, self.tui_to_group)
            self._local.matcher = matcher
        return matcher

    def _key(self, entity: dict) -> str:
        raw_key = "||".join([
            str(entity.get("surface_form") or "").strip().lower(),
            str(entity.get("normalized_form") or "").strip().lower(),
            str(entity.get("semantic_group") or "").strip().upper(),
        ])
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key[:2], f"{key}.json")

    def _lock_for(self, key: str) -> threading.Lock:
        with self._gate:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _load_cached(self, path: str) -> Optional[dict]:
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def match(self, entity: dict) -> dict:
        key = self._key(entity)
        path = self._cache_path(key)
        cached = self._load_cached(path)
        if cached is not None:
            with self._gate:
                self.stats["cache_hits"] += 1
            return cached

        lock = self._lock_for(key)
        with lock:
            cached = self._load_cached(path)
            if cached is not None:
                with self._gate:
                    self.stats["cache_hits"] += 1
                return cached

            try:
                result = self._matcher().match_entity(entity)
            except Exception as exc:
                logger.warning("UMLS match failed for %r: %s", entity.get("surface_form"), exc)
                result = {
                    "entity": entity,
                    "matched": False,
                    "matches": [],
                    "match_type": "error",
                }
                with self._gate:
                    self.stats["match_errors"] += 1

            _atomic_write_json(path, result)
            with self._gate:
                self.stats["cache_misses"] += 1
            return result

    def match_many(self, entities: list[dict]) -> list[dict]:
        return [self.match(entity) for entity in entities]


class FileBackedSubgraphCache:
    """1-hop subgraph cache with one-cache-file-per-CUI."""

    def __init__(self, api_key: str, cache_dir: str):
        self.api_key = api_key
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._gate = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._local = threading.local()
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "fetch_errors": 0,
        }

    def _client(self) -> UMLSClient:
        client = getattr(self._local, "client", None)
        if client is None:
            client = UMLSClient(api_key=self.api_key)
            self._local.client = client
        return client

    def _cache_path(self, cui: str) -> str:
        return os.path.join(self.cache_dir, cui[:3], f"{cui}.json")

    def _lock_for(self, cui: str) -> threading.Lock:
        with self._gate:
            if cui not in self._locks:
                self._locks[cui] = threading.Lock()
            return self._locks[cui]

    def _load_cached(self, path: str) -> Optional[list[dict]]:
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, list) else []

    def get_triples(self, cui: str, seed_name: str) -> list[dict]:
        path = self._cache_path(cui)
        cached = self._load_cached(path)
        if cached is not None:
            with self._gate:
                self.stats["cache_hits"] += 1
            return cached

        lock = self._lock_for(cui)
        with lock:
            cached = self._load_cached(path)
            if cached is not None:
                with self._gate:
                    self.stats["cache_hits"] += 1
                return cached

            try:
                triples = build_1hop_subgraph(self._client(), cui, seed_name=seed_name)
            except Exception as exc:
                logger.warning("Subgraph fetch failed for %s: %s", cui, exc)
                triples = []
                with self._gate:
                    self.stats["fetch_errors"] += 1

            _atomic_write_json(path, triples)
            with self._gate:
                self.stats["cache_misses"] += 1
            return triples


def _resolve_input_files(args: argparse.Namespace) -> list[str]:
    if args.input_files:
        return [os.path.join(args.input_dir, name) for name in args.input_files]
    return [os.path.join(args.input_dir, name) for name in DEFAULT_INPUT_FILES]


def _iter_jsonl_records(path: str, max_samples: Optional[int], start_index: int):
    yielded = 0
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if seen < start_index:
                seen += 1
                continue
            if max_samples is not None and yielded >= max_samples:
                break
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc
            seen += 1
            yielded += 1
            yield raw


def _sample_id(raw: dict, sample_index: int) -> str:
    value = raw.get("id")
    if value is None or str(value).strip() == "":
        return f"sample_{sample_index}"
    return str(value)


def _extract_main_entities(raw: dict, sample_id: str) -> list[dict]:
    extracted = raw.get("extracted_info")
    if not isinstance(extracted, dict) or not extracted:
        return []

    sample_info = extracted.get(sample_id)
    if sample_info is None and len(extracted) == 1:
        sample_info = next(iter(extracted.values()))
    if not isinstance(sample_info, dict):
        return []

    entities = sample_info.get("entities")
    if not isinstance(entities, list):
        return []
    return entities


def _dedup_main_entities(entities: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for entity in entities:
        key = "||".join([
            str(entity.get("surface_form") or "").strip().lower(),
            str(entity.get("normalized_form") or "").strip().lower(),
            str(entity.get("semantic_group") or "").strip().upper(),
        ])
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def _aggregate_umls_entities(match_results: list[dict]) -> list[dict]:
    aggregated: dict[str, dict] = {}
    order: list[str] = []
    seen_sources: dict[str, set[str]] = {}

    for result in match_results:
        if not result.get("matched"):
            continue
        entity = result.get("entity") or {}
        source_key = "||".join([
            str(entity.get("surface_form") or "").strip(),
            str(entity.get("normalized_form") or "").strip(),
            str(entity.get("semantic_group") or "").strip(),
        ])
        for match in result.get("matches", []):
            cui = str(match.get("cui") or "").strip()
            if not cui:
                continue
            if cui not in aggregated:
                aggregated[cui] = {
                    "cui": cui,
                    "name": match.get("name", "") or cui,
                    "root_sources": [],
                    "match_types": [],
                    "matched_queries": [],
                    "source_entities": [],
                    "semantic_types": match.get("semantic_types") or [],
                }
                seen_sources[cui] = set()
                order.append(cui)

            item = aggregated[cui]
            _append_unique(item["root_sources"], match.get("root_source", ""))
            _append_unique(item["match_types"], match.get("match_type", ""))
            _append_unique(item["matched_queries"], match.get("match_query", ""))
            if source_key not in seen_sources[cui]:
                seen_sources[cui].add(source_key)
                item["source_entities"].append({
                    "surface_form": entity.get("surface_form", ""),
                    "normalized_form": entity.get("normalized_form", ""),
                    "semantic_group": entity.get("semantic_group", ""),
                })

    return [aggregated[cui] for cui in order]


def _build_sub_docs(
    main_entities: list[dict],
    umls_entities: list[dict],
    umls_triples: list[dict],
) -> dict:
    return {
        "meta_data": {
            "main_entity_count": len(main_entities),
            "umls_entity_count": len(umls_entities),
            "umls_triple_count": len(umls_triples),
        },
        "main_entity": main_entities,
        "umls_entity": umls_entities,
        "umls_triple": umls_triples,
    }


def _deduplicate_triples_silent(triples: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for triple in triples:
        key = (
            str(triple.get("head_cui") or ""),
            str(triple.get("relation") or ""),
            str(triple.get("tail_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(triple)
    return unique


def _process_sample(
    raw: dict,
    sample_index: int,
    matcher: FileBackedUMLSMatcher,
    subgraph_cache: FileBackedSubgraphCache,
) -> tuple[dict, dict]:
    sample_id = _sample_id(raw, sample_index)
    main_entities = _dedup_main_entities(_extract_main_entities(raw, sample_id))

    match_results = matcher.match_many(main_entities)
    umls_entities = _aggregate_umls_entities(match_results)

    triples: list[dict] = []
    for umls_entity in umls_entities:
        triples.extend(
            subgraph_cache.get_triples(
                cui=umls_entity["cui"],
                seed_name=umls_entity.get("name", "") or umls_entity["cui"],
            )
        )
    umls_triples = _deduplicate_triples_silent(triples) if triples else []

    raw["sub_docs"] = {
        sample_id: _build_sub_docs(main_entities, umls_entities, umls_triples)
    }
    stats = {
        "sample_id": sample_id,
        "main_entities": len(main_entities),
        "umls_entities": len(umls_entities),
        "umls_triples": len(umls_triples),
    }
    return raw, stats


def _log_progress(
    *,
    processed: int,
    file_total_entities: int,
    file_total_umls_entities: int,
    file_total_umls_triples: int,
    matcher: FileBackedUMLSMatcher,
    subgraph_cache: FileBackedSubgraphCache,
    file_label: str,
) -> None:
    avg_entities = (file_total_entities / processed) if processed else 0.0
    avg_umls = (file_total_umls_entities / processed) if processed else 0.0
    avg_triples = (file_total_umls_triples / processed) if processed else 0.0
    logger.info(
        "%s progress: samples=%d avg_main_entity=%.2f avg_umls_entity=%.2f "
        "avg_umls_triple=%.2f entity_cache(hit=%d miss=%d) subgraph_cache(hit=%d miss=%d)",
        file_label,
        processed,
        avg_entities,
        avg_umls,
        avg_triples,
        matcher.stats["cache_hits"],
        matcher.stats["cache_misses"],
        subgraph_cache.stats["cache_hits"],
        subgraph_cache.stats["cache_misses"],
    )


def _drain_batch(
    batch: list[tuple[int, dict]],
    executor: Optional[ThreadPoolExecutor],
    matcher: FileBackedUMLSMatcher,
    subgraph_cache: FileBackedSubgraphCache,
) -> list[tuple[dict, dict]]:
    if not batch:
        return []

    results: list[Optional[tuple[dict, dict]]] = [None] * len(batch)
    if executor is None:
        for idx, (sample_index, raw) in enumerate(batch):
            results[idx] = _process_sample(raw, sample_index, matcher, subgraph_cache)
        return [r for r in results if r is not None]

    futures: dict[Future, int] = {}
    for idx, (sample_index, raw) in enumerate(batch):
        future = executor.submit(_process_sample, raw, sample_index, matcher, subgraph_cache)
        futures[future] = idx

    for future in as_completed(futures):
        idx = futures[future]
        results[idx] = future.result()

    return [r for r in results if r is not None]


def _process_file(
    input_path: str,
    output_path: str,
    args: argparse.Namespace,
    matcher: FileBackedUMLSMatcher,
    subgraph_cache: FileBackedSubgraphCache,
) -> None:
    logger.info("Processing %s -> %s", input_path, output_path)

    processed = 0
    total_main_entities = 0
    total_umls_entities = 0
    total_umls_triples = 0
    batch: list[tuple[int, dict]] = []
    executor = None
    if args.workers > 1:
        executor = ThreadPoolExecutor(max_workers=args.workers)

    try:
        with open(output_path, "w", encoding="utf-8", buffering=1024 * 1024) as out_f:
            for sample_index, raw in enumerate(
                _iter_jsonl_records(
                    input_path,
                    max_samples=args.max_samples,
                    start_index=args.start_index,
                ),
                start=args.start_index,
            ):
                batch.append((sample_index, raw))
                if len(batch) < args.batch_size:
                    continue

                for updated_raw, stats in _drain_batch(batch, executor, matcher, subgraph_cache):
                    out_f.write(json.dumps(updated_raw, **_JSON_DUMP_KWARGS))
                    out_f.write("\n")
                    processed += 1
                    total_main_entities += stats["main_entities"]
                    total_umls_entities += stats["umls_entities"]
                    total_umls_triples += stats["umls_triples"]
                    if processed % args.progress_interval == 0:
                        out_f.flush()
                        _log_progress(
                            processed=processed,
                            file_total_entities=total_main_entities,
                            file_total_umls_entities=total_umls_entities,
                            file_total_umls_triples=total_umls_triples,
                            matcher=matcher,
                            subgraph_cache=subgraph_cache,
                            file_label=os.path.basename(input_path),
                        )
                batch.clear()

            if batch:
                for updated_raw, stats in _drain_batch(batch, executor, matcher, subgraph_cache):
                    out_f.write(json.dumps(updated_raw, **_JSON_DUMP_KWARGS))
                    out_f.write("\n")
                    processed += 1
                    total_main_entities += stats["main_entities"]
                    total_umls_entities += stats["umls_entities"]
                    total_umls_triples += stats["umls_triples"]
                batch.clear()
            out_f.flush()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    logger.info(
        "Completed %s: samples=%d main_entities=%d umls_entities=%d umls_triples=%d",
        os.path.basename(input_path),
        processed,
        total_main_entities,
        total_umls_entities,
        total_umls_triples,
    )
    _log_progress(
        processed=processed,
        file_total_entities=total_main_entities,
        file_total_umls_entities=total_umls_entities,
        file_total_umls_triples=total_umls_triples,
        matcher=matcher,
        subgraph_cache=subgraph_cache,
        file_label=os.path.basename(input_path),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build UMLS sub-docs from extract_info.py JSONL outputs.",
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=None,
        help="Optional subset of extracted JSONL filenames to process",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.progress_interval < 1:
        raise ValueError("--progress-interval must be >= 1")

    os.makedirs(args.output_dir, exist_ok=True)
    input_files = _resolve_input_files(args)
    for input_path in input_files:
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

    cfg = _build_cfg(
        args,
        dataset_path=input_files[0],
        output_dir=args.output_dir,
    )
    matcher = FileBackedUMLSMatcher(
        cfg,
        cache_dir=os.path.join(cfg.cache_dir, "umls_match_files"),
    )
    subgraph_cache = FileBackedSubgraphCache(
        api_key=cfg.umls_api_key,
        cache_dir=os.path.join(cfg.cache_dir, "subgraph_files"),
    )

    logger.info(
        "Starting sub-doc build: files=%d workers=%d batch_size=%d output_dir=%s",
        len(input_files),
        args.workers,
        args.batch_size,
        args.output_dir,
    )
    for input_path in input_files:
        output_path = os.path.join(args.output_dir, os.path.basename(input_path))
        _process_file(
            input_path=input_path,
            output_path=output_path,
            args=args,
            matcher=matcher,
            subgraph_cache=subgraph_cache,
        )

    logger.info(
        "Finished sub-doc build. entity_cache(hit=%d miss=%d err=%d) "
        "subgraph_cache(hit=%d miss=%d err=%d)",
        matcher.stats["cache_hits"],
        matcher.stats["cache_misses"],
        matcher.stats["match_errors"],
        subgraph_cache.stats["cache_hits"],
        subgraph_cache.stats["cache_misses"],
        subgraph_cache.stats["fetch_errors"],
    )


if __name__ == "__main__":
    main()
