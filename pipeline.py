"""
Medical KG Pipeline — Unified Entry Point

Single CLI dispatching all five stages via subcommands:
  Stage 0: CREST parsing & recommendation/context extraction
  Stage 1: Entity candidate extraction (LLM)
  Stage 2: UMLS layer construction (matching + triple extraction)
  Stage 3: Condition augmentation
  Stage 4: Neo4j knowledge-graph construction

Usage:
    # individual stages
    python pipeline.py stage0 --xml-dir ./crest/xml --primary-dir ./crest/primary
    python pipeline.py stage1 --openai-key sk-...
    python pipeline.py stage2 --umls-key ...
    python pipeline.py stage3 --max-triples 10
    python pipeline.py stage4 --neo4j-password ... [--clear]

    # full pipeline (stages 0-3 by default; --end-stage 4 to also build the KG)
    python pipeline.py all --umls-key ... --openai-key ...
    python pipeline.py all --end-stage 4 --neo4j-password ...

    # partial range or test slice
    python pipeline.py all --start-stage 1 --end-stage 2
    python pipeline.py all --max-guideline-text 10 --max-triples 50

    # backgrounded
    nohup python -u pipeline.py stage3 --output-dir ./output \\
        > ./output/stage3_stdout.log 2>&1 &
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from itertools import islice
from typing import Any

import condition_augmenter
import config

from cli_utils import load_json, require_files, save_json, setup_logging
from crest_parser import extract_from_both_sources
from dataset.Pubmed.get_pubmed_data import extract_from_sqlite

from entity_extractor import (
    iter_entity_extraction_batches,
    merge_unique_entities,
)
from entity_matcher import match_entities_batch
from semantic_types import load_semantic_groups_from_file
from subgraph_builder import build_subgraphs_batch, deduplicate_triples
from UMLS_KG.umls_client import UMLSClient

logger = logging.getLogger(__name__)
STAGE0_DOCUMENTS_JSONL_FILE = "stage0_documents.jsonl"
PUBMED_STAGE0_BATCH_SIZE = 1000


def _stage0_documents_jsonl_path(output_dir: str) -> str:
    return os.path.join(output_dir, STAGE0_DOCUMENTS_JSONL_FILE)


def _stage0_dataset_name(doc: dict) -> str:
    """Infer the source dataset name from a Stage 0 document row."""
    doc_id = str(doc.get("db_guideline_id", "")).strip()
    if doc_id.startswith("CREST_"):
        return "CREST"
    if doc_id.startswith("PUBMED_"):
        return "PUBMED"
    return "UNKNOWN"


def _iter_crest_documents(
    db_name: str,
    max_guideline_text: int = None,
    xml_dir: str = None,
    primary_dir: str = None,
):
    """Yield grouped CREST guideline documents."""
    grouped_documents: dict[str, dict[str, Any]] = {}
    guideline_text = extract_from_both_sources(
        xml_dir=xml_dir or config.CREST_XML_DIR,
        primary_dir=primary_dir or config.CREST_PRIMARY_DIR,
        max_guideline_text=max_guideline_text,
    )

    if not guideline_text:
        logger.error("No recommendations extracted. Check CREST paths.")
        sys.exit(1)

    for sentence_index, rec in enumerate(guideline_text, start=1):
        guideline_id = str(rec.get("guideline_id", "")).strip() or f"doc_{sentence_index}"
        db_guideline_id = f"CREST_{guideline_id}"
        raw_text = (rec.get("text") or rec.get("guideline_context") or "").strip()
        strength = (rec.get("strength") or "").strip()

        if db_guideline_id not in grouped_documents:
            grouped_documents[db_guideline_id] = {
                "db_guideline_id": db_guideline_id,
                "guideline_context": rec.get("guideline_context", ""),
                "raw_texts": [],
                "sentence_count": 0,
            }

        doc_entry = grouped_documents[db_guideline_id]
        doc_entry["raw_texts"].append(f"{strength} : {raw_text}")
        doc_entry["sentence_count"] += 1

    yield from grouped_documents.values()


def _iter_pubmed_documents(
    max_guideline_text: int = None,
    pubmed_sqlite_path: str = None,
):
    """Yield one document per PubMed abstract row from SQLite."""
    db_path = pubmed_sqlite_path or config.PUBMED_SQLITE_PATH
    for pubmed_batch in extract_from_sqlite(
        db_path=db_path,
        max_guideline_text=max_guideline_text,
        batch_size=PUBMED_STAGE0_BATCH_SIZE,
    ):
        for rec in pubmed_batch:
            pmid = str(rec.get("pmid", "")).strip()
            title = (rec.get("title") or "").strip()
            abstract = (rec.get("abstract") or "").strip()
            merged_text = f"{title}\n\n{abstract}" if title and abstract else title or abstract
            if not merged_text:
                continue

            yield {
                "db_guideline_id": f"PUBMED_{pmid}" if pmid else "",
                "guideline_context": "",
                "raw_texts": [merged_text],
                "sentence_count": 1,
            }


def _iter_stage0_documents(
    db_name: list[str],
    max_guideline_text: int = None,
    xml_dir: str = None,
    primary_dir: str = None,
    pubmed_sqlite_path: str = None,
):
    """Yield Stage 0 documents without materializing the whole corpus."""
    db_name = db_name or ["PUBMED"]
    if "CREST" in db_name:
        yield from _iter_crest_documents(
            db_name=db_name,
            max_guideline_text=max_guideline_text,
            xml_dir=xml_dir,
            primary_dir=primary_dir,
        )

    if "PUBMED" in db_name:
        yield from _iter_pubmed_documents(
            max_guideline_text=max_guideline_text,
            pubmed_sqlite_path=pubmed_sqlite_path,
        )


def _write_stage0_documents(
    output_dir: str,
    db_name: list[str],
    max_guideline_text: int = None,
    xml_dir: str = None,
    primary_dir: str = None,
    pubmed_sqlite_path: str = None,
) -> tuple[str, int, int, dict[str, dict[str, Any]] | None, dict[str, dict[str, Any]]]:
    """Persist Stage 0 documents to JSONL and optionally inline small sources."""
    documents_jsonl_path = _stage0_documents_jsonl_path(output_dir)
    inline_documents_by_id = {} if "PUBMED" not in (db_name or []) else None
    total_document_count = 0
    total_sentence_count = 0
    dataset_examples: dict[str, dict[str, Any]] = {}

    with open(documents_jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for doc in _iter_stage0_documents(
            db_name=db_name,
            max_guideline_text=max_guideline_text,
            xml_dir=xml_dir,
            primary_dir=primary_dir,
            pubmed_sqlite_path=pubmed_sqlite_path,
        ):
            doc_id = str(doc.get("db_guideline_id", "")).strip()
            if not doc_id:
                continue

            jsonl_file.write(json.dumps(doc, ensure_ascii=False, default=str))
            jsonl_file.write("\n")

            total_document_count += 1
            total_sentence_count += doc.get("sentence_count", len(doc.get("raw_texts") or []))

            if inline_documents_by_id is not None:
                inline_documents_by_id[doc_id] = doc

            dataset_name = _stage0_dataset_name(doc)
            if dataset_name not in dataset_examples:
                dataset_examples[dataset_name] = dict(doc)

            if total_document_count % PUBMED_STAGE0_BATCH_SIZE == 0:
                logger.info(
                    f"  Stage 0 progress: wrote {total_document_count} documents "
                    f"({total_sentence_count} sentences)"
                )

    if total_document_count == 0:
        logger.error(f"No records extracted for db_name={db_name}")
        sys.exit(1)

    return (
        documents_jsonl_path,
        total_document_count,
        total_sentence_count,
        inline_documents_by_id,
        dataset_examples,
    )


def _iter_documents_from_jsonl(documents_jsonl_path: str):
    with open(documents_jsonl_path, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _resolve_stage0_documents_jsonl_path(stage0_data: dict, output_dir: str) -> str | None:
    """Resolve the Stage 0 JSONL sidecar path recorded in stage0_recommendations.json."""
    documents_jsonl_path = stage0_data.get("documents_jsonl")
    if not documents_jsonl_path:
        return None

    if not os.path.isabs(documents_jsonl_path):
        documents_jsonl_path = os.path.join(output_dir, documents_jsonl_path)

    return documents_jsonl_path


def _load_stage0_documents_input(stage0_data: dict, output_dir: str):
    documents_jsonl_path = _resolve_stage0_documents_jsonl_path(stage0_data, output_dir)
    if documents_jsonl_path:
        if not os.path.isfile(documents_jsonl_path):
            logger.error("Stage 0 documents JSONL referenced by stage0_recommendations.json was not found:")
            logger.error(f"  - documents_jsonl: {documents_jsonl_path}")
            logger.error("Re-run `python pipeline.py stage0` to regenerate the Stage 0 outputs.")
            sys.exit(1)
        return _iter_documents_from_jsonl(documents_jsonl_path)

    documents_by_id: dict[str, dict[str, Any]] = {}
    documents_by_id = stage0_data.get("documents_by_id", {})
    if isinstance(documents_by_id, dict):
        return documents_by_id.values()
    if documents_by_id:
        return documents_by_id

    legacy_records = stage0_data.get("records") or stage0_data.get("recommendations", [])
    return legacy_records or []


# ──────────────────────────────────────────────────────────────────────
# Stage 0: CREST Parsing & Recommendation/Context Extraction
# ──────────────────────────────────────────────────────────────────────

def run_stage0(
    db_name: list = None,
    output_dir: str = None,
    max_guideline_text: int = None,
    xml_dir: str = None,
    primary_dir: str = None,
    pubmed_sqlite_path: str = None,
) -> list[dict]:
    """Stage 0: parse CREST → save recommendations file. Returns the list."""
    output_dir = output_dir or config.OUTPUT_DIR
    db_name = db_name or ["PUBMED"]
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STAGE 0: Guideline parsing & extraction")
    logger.info("=" * 60)

    start = time.time()

    (
        documents_jsonl_path,
        total_document_count,
        total_sentence_count,
        inline_documents_by_id,
        dataset_examples,
    ) = _write_stage0_documents(
        output_dir=output_dir,
        db_name=db_name,
        max_guideline_text=max_guideline_text,
        xml_dir=xml_dir,
        primary_dir=primary_dir,
        pubmed_sqlite_path=pubmed_sqlite_path,
    )

    elapsed = time.time() - start
    output_path = os.path.join(output_dir, config.OUTPUT_RECOMMENDATIONS_FILE)

    output_data = {
        "metadata": {
            "stage": 0,
            "db_name": db_name,
            "total_document_count": total_document_count,
            "total_sentence_count": total_sentence_count,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        },
        "documents_jsonl": os.path.basename(documents_jsonl_path),
        "stage0_documents_jsonl_examples": dataset_examples,
    }
    if inline_documents_by_id is not None:
        output_data["documents_by_id"] = inline_documents_by_id

    save_json(output_data, output_path)

    logger.info(f"Stage 0 complete in {elapsed:.1f}s")
    logger.info(f"  DB name: {db_name}")
    logger.info(f"  Raw-text records: {total_sentence_count}")
    logger.info(f"  Document IDs: {total_document_count}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Documents JSONL: {documents_jsonl_path}")
    logger.info(f"Next: python pipeline.py stage1 --output-dir {output_dir}")
    return inline_documents_by_id or []


# ──────────────────────────────────────────────────────────────────────
# Stage 1: Entity Candidate Extraction (LLM)
# ──────────────────────────────────────────────────────────────────────

def run_stage1(
    output_dir: str = None,
    openai_api_key: str = None,
    max_workers: int = None,
    max_guideline_docs: int = None,
    progress_interval: int = 10,
) -> tuple[int, dict]:
    """Stage 1: load Stage 0 documents, extract entities, and deduplicate incrementally."""
    output_dir = output_dir or config.OUTPUT_DIR

    if openai_api_key:
        config.OPENAI_API_KEY = openai_api_key

    recs_path = os.path.join(output_dir, config.OUTPUT_RECOMMENDATIONS_FILE)
    require_files(
        {"stage0_recommendations": recs_path},
        hint="Run `python pipeline.py stage0` first.",
    )

    logger.info("=" * 60)
    logger.info("STAGE 1: Entity Candidate Extraction")
    logger.info("=" * 60)

    recs_data = load_json(recs_path)
    metadata = recs_data.get("metadata", {})
    input_document_count = metadata.get("total_document_count", 0)
    input_sentence_count = metadata.get("total_sentence_count", 0)
    documents_input = _load_stage0_documents_input(recs_data, output_dir)

    documents_jsonl_path = _resolve_stage0_documents_jsonl_path(recs_data, output_dir)
    if documents_jsonl_path:
        logger.info(f"  Using Stage 0 documents JSONL: {documents_jsonl_path}")

    if max_guideline_docs:
        documents_input = list(islice(documents_input, max_guideline_docs))
        input_document_count = len(documents_input)
        input_sentence_count = sum(
            len(doc.get("raw_texts") or [])
            for doc in documents_input
            if isinstance(doc, dict)
        )
        logger.info(f"  Limited to {max_guideline_docs} documents (test mode)")
        logger.info(f"  Loaded {input_document_count} documents from Stage 0")
        logger.info(f"  Loaded {input_sentence_count} raw-text sentences from Stage 0")
    else:
        logger.info(f"  Loaded {input_document_count} documents from Stage 0")
        logger.info(f"  Loaded {input_sentence_count} raw-text sentences from Stage 0")
        

    start = time.time()
    total_raw_entities = 0
    unique_entities: dict = {}
    for batch_entities, _batch_sentence_count, _batch_failed_count in iter_entity_extraction_batches(
        documents_input,
        progress_interval=progress_interval,
        max_workers=max_workers,
    ):
        total_raw_entities += len(batch_entities)
        merge_unique_entities(unique_entities, batch_entities)

    elapsed = time.time() - start

    output_path = os.path.join(output_dir, config.OUTPUT_ENTITIES_FILE)
    save_json(
        {
            "metadata": {
                "stage": 1,
                "input_documents": input_document_count,
                "input_sentences": input_sentence_count,
                "total_raw_entities": total_raw_entities,
                "total_unique_entities": len(unique_entities),
                "elapsed_seconds": round(elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            },
            "entities": list(unique_entities.values()),
        },
        output_path,
    )

    logger.info(f"Stage 1 complete in {elapsed:.1f}s")
    logger.info(f"  Raw entities: {total_raw_entities}")
    logger.info(f"  Unique entities: {len(unique_entities)}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"Next: python pipeline.py stage2 --output-dir {output_dir}")
    return total_raw_entities, unique_entities


# ──────────────────────────────────────────────────────────────────────
# Stage 2: UMLS Layer Construction (Triple Extraction)
# ──────────────────────────────────────────────────────────────────────

def run_stage2(
    output_dir: str = None,
    umls_api_key: str = None,
    semantic_groups_file: str = None,
    max_workers: int = None,
    progress_interval: int = 50,
) -> tuple[list[dict], dict, list[dict]]:
    """Stage 2: load Stage 1 entities, run UMLS matching + subgraph collection."""
    output_dir = output_dir or config.OUTPUT_DIR

    if umls_api_key:
        config.UMLS_API_KEY = umls_api_key

    ent_path = os.path.join(output_dir, config.OUTPUT_ENTITIES_FILE)
    require_files(
        {"stage1_entity_candidates": ent_path},
        hint="Run `python pipeline.py stage1` first.",
    )

    logger.info("=" * 60)
    logger.info("STAGE 2: UMLS Layer Construction")
    logger.info("=" * 60)

    # Load Stage 1 output
    ent_data = load_json(ent_path)
    entities_list = ent_data.get("entities", [])
    logger.info(f"  Loaded {len(entities_list)} entity candidates from Stage 1")

    # match_entities_batch only iterates .values(); reconstruct a dict keyed by
    # normalized_form to mirror the in-memory shape used in deduplicate_entities.
    unique_entities = {
        ent.get("normalized_form", ent.get("surface_form", "")).lower().strip(): ent
        for ent in entities_list
    }

    # Semantic group mapping (TUI → group)
    sgroups_file = semantic_groups_file or config.SEMANTIC_GROUPS_FILE
    tui_to_group, _ = load_semantic_groups_from_file(sgroups_file)
    logger.info(f"  Loaded {len(tui_to_group)} TUI → semantic-group mappings")

    umls_client = UMLSClient()

    # ── Stage 2-A: Entity → UMLS CUI matching ──
    logger.info("Stage 2-A: Entity → UMLS CUI matching")
    start_a = time.time()
    match_results, matched_cuis = match_entities_batch(
        unique_entities, umls_client, tui_to_group,
        progress_interval=progress_interval,
        max_workers=max_workers,
    )
    elapsed_a = time.time() - start_a

    matched_count = sum(1 for r in match_results if r["matched"])
    matched_path = os.path.join(output_dir, config.OUTPUT_MATCHED_FILE)
    save_json(
        {
            "metadata": {
                "stage": "2-A",
                "total_match_results": len(match_results),
                "matched_count": matched_count,
                "match_rate_percent": round(
                    matched_count / max(len(match_results), 1) * 100, 1
                ),
                "total_unique_cuis": len(matched_cuis),
                "umls_api_requests_so_far": umls_client.request_count,
                "elapsed_seconds": round(elapsed_a, 1),
                "timestamp": datetime.now().isoformat(),
            },
            "match_results": match_results,
            "matched_cuis": matched_cuis,
        },
        matched_path,
    )

    # ── Stage 2-B: 1-hop subgraph collection ──
    logger.info("Stage 2-B: 1-hop subgraph collection")
    start_b = time.time()
    all_triples = build_subgraphs_batch(
        umls_client, matched_cuis,
        progress_interval=progress_interval,
        max_workers=max_workers,
    )
    unique_triples = deduplicate_triples(all_triples)
    elapsed_b = time.time() - start_b

    triples_path = os.path.join(output_dir, config.OUTPUT_TRIPLES_FILE)
    save_json(
        {
            "metadata": {
                "stage": "2-B",
                "raw_triples_before_dedup": len(all_triples),
                "total_triples": len(unique_triples),
                "umls_api_requests_total": umls_client.request_count,
                "elapsed_seconds": round(elapsed_b, 1),
                "timestamp": datetime.now().isoformat(),
            },
            "triples": unique_triples,
        },
        triples_path,
    )

    elapsed = elapsed_a + elapsed_b
    logger.info(f"Stage 2 complete in {elapsed:.1f}s "
                f"(2-A: {elapsed_a:.1f}s, 2-B: {elapsed_b:.1f}s)")
    logger.info(f"  Match rate: {matched_count}/{len(match_results)} "
                f"({matched_count / max(len(match_results), 1) * 100:.1f}%)")
    logger.info(f"  Unique CUIs: {len(matched_cuis)}")
    logger.info(f"  Unique triples: {len(unique_triples)}")
    logger.info(f"  UMLS API requests: {umls_client.request_count}")
    logger.info(f"Next: python pipeline.py stage3 --output-dir {output_dir}")
    return match_results, matched_cuis, unique_triples


# ──────────────────────────────────────────────────────────────────────
# Stage 3: Condition Augmentation
# ──────────────────────────────────────────────────────────────────────

def run_stage3(
    output_dir: str = None,
    openai_api_key: str = None,
    batch_size: int = None,
    max_triples: int = None,
    max_workers: int = None,
) -> list[dict]:
    """Stage 3: load Stage 0/1/2 outputs, augment triples with conditions."""
    output_dir = output_dir or config.OUTPUT_DIR

    if openai_api_key:
        config.OPENAI_API_KEY = openai_api_key

    required = {
        "stage0_recommendations":
            os.path.join(output_dir, config.OUTPUT_RECOMMENDATIONS_FILE),
        "stage1_entity_candidates":
            os.path.join(output_dir, config.OUTPUT_ENTITIES_FILE),
        "stage2_umls_matched":
            os.path.join(output_dir, config.OUTPUT_MATCHED_FILE),
        "stage2_umls_layer_triples":
            os.path.join(output_dir, config.OUTPUT_TRIPLES_FILE),
    }
    require_files(
        required,
        hint="Run `python pipeline.py stage0` through `stage2` first.",
    )

    logger.info("Loading Stage 0/1/2 outputs...")
    recommendations = load_json(required["stage0_recommendations"]).get("recommendations", [])
    entities = load_json(required["stage1_entity_candidates"]).get("entities", [])
    match_results = load_json(required["stage2_umls_matched"]).get("match_results", [])
    triples = load_json(required["stage2_umls_layer_triples"]).get("triples", [])

    logger.info(f"  Recommendations: {len(recommendations)}")
    logger.info(f"  Entity candidates: {len(entities)}")
    logger.info(f"  Match results: {len(match_results)}")
    logger.info(f"  UMLS layer triples: {len(triples)}")

    if max_triples:
        triples = triples[:max_triples]
        logger.info(f"  Limited to {max_triples} triples (test mode)")

    start_time = time.time()
    augmented_triples = condition_augmenter.run_stage3(
        triples=triples,
        recommendations=recommendations,
        entities=entities,
        match_results=match_results,
        batch_size=batch_size,
        max_workers=max_workers,
    )
    elapsed = time.time() - start_time

    # ── Aggregate stats ──
    total_with_cond = sum(1 for t in augmented_triples if t.get("has_conditions"))
    total_cond = sum(len(t.get("conditions", [])) for t in augmented_triples)
    total_parse_failed = sum(1 for t in augmented_triples if t.get("parse_failed"))

    cond_type_dist: dict = {}
    cond_subtype_dist: dict = {}
    cond_qualifies_dist: dict = {}
    cond_status_dist: dict = {}  # legacy medication_history.status / backward compatibility
    cond_clinical_status_dist: dict = {}
    cond_verification_status_dist: dict = {}
    cond_severity_dist: dict = {}
    cond_body_site_dist: dict = {}
    cond_temporal_relation_dist: dict = {}
    cond_frequency_dist: dict = {}
    cond_route_dist: dict = {}
    cond_range_count = 0
    cond_value_operator_dist: dict = {}
    for t in augmented_triples:
        for c in t.get("conditions", []):
            ct = c.get("type", "unknown")
            cond_type_dist[ct] = cond_type_dist.get(ct, 0) + 1

            st = c.get("subtype")
            if st:
                cond_subtype_dist[st] = cond_subtype_dist.get(st, 0) + 1

            q = c.get("qualifies")
            if q:
                cond_qualifies_dist[q] = cond_qualifies_dist.get(q, 0) + 1

            status = c.get("status")
            if status:
                cond_status_dist[status] = cond_status_dist.get(status, 0) + 1

            clinical_status = c.get("clinical_status")
            if clinical_status:
                cond_clinical_status_dist[clinical_status] = cond_clinical_status_dist.get(clinical_status, 0) + 1

            verification_status = c.get("verification_status")
            if verification_status:
                cond_verification_status_dist[verification_status] = cond_verification_status_dist.get(verification_status, 0) + 1

            severity = c.get("severity")
            if severity:
                cond_severity_dist[severity] = cond_severity_dist.get(severity, 0) + 1

            body_site = c.get("body_site")
            if body_site:
                cond_body_site_dist[body_site] = cond_body_site_dist.get(body_site, 0) + 1

            temporal_relation = c.get("temporal_relation")
            if temporal_relation:
                cond_temporal_relation_dist[temporal_relation] = cond_temporal_relation_dist.get(temporal_relation, 0) + 1

            frequency = c.get("frequency")
            if frequency:
                cond_frequency_dist[frequency] = cond_frequency_dist.get(frequency, 0) + 1

            route = c.get("route")
            if route:
                cond_route_dist[route] = cond_route_dist.get(route, 0) + 1

            if c.get("value_min") is not None or c.get("value_max") is not None:
                cond_range_count += 1

            vo = c.get("value_operator")
            if vo:
                cond_value_operator_dist[vo] = cond_value_operator_dist.get(vo, 0) + 1

    strength_dist: dict = {}
    for t in augmented_triples:
        s = t.get("recommendation_strength")
        if s:
            strength_dist[s] = strength_dist.get(s, 0) + 1

    output_data = {
        "metadata": {
            "stage": 3,
            "total_triples": len(augmented_triples),
            "triples_with_conditions": total_with_cond,
            "triples_without_conditions": len(augmented_triples) - total_with_cond,
            "triples_parse_failed": total_parse_failed,
            "total_conditions": total_cond,
            "condition_type_distribution": cond_type_dist,
            "condition_subtype_distribution": cond_subtype_dist,
            "condition_qualifies_distribution": cond_qualifies_dist,
            "condition_status_distribution": cond_status_dist,
            "condition_clinical_status_distribution": cond_clinical_status_dist,
            "condition_verification_status_distribution": cond_verification_status_dist,
            "condition_severity_distribution": cond_severity_dist,
            "condition_body_site_distribution": cond_body_site_dist,
            "condition_temporal_relation_distribution": cond_temporal_relation_dist,
            "condition_frequency_distribution": cond_frequency_dist,
            "condition_route_distribution": cond_route_dist,
            "condition_numeric_range_count": cond_range_count,
            "condition_value_operator_distribution": cond_value_operator_dist,
            "recommendation_strength_distribution": strength_dist,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        },
        "triples": augmented_triples,
    }

    augmented_path = os.path.join(output_dir, config.OUTPUT_AUGMENTED_TRIPLES_FILE)
    save_json(output_data, augmented_path)

    logger.info(f"Stage 3 complete in {elapsed:.1f}s")
    logger.info(f"  Output: {augmented_path}")
    logger.info(f"  Condition types: {cond_type_dist}")
    logger.info(f"  Condition subtypes: {cond_subtype_dist}")
    logger.info(f"  Condition qualifies: {cond_qualifies_dist}")
    logger.info(f"  Condition medication/legacy statuses: {cond_status_dist}")
    logger.info(f"  Condition clinical statuses: {cond_clinical_status_dist}")
    logger.info(f"  Condition verification statuses: {cond_verification_status_dist}")
    logger.info(f"  Condition severities: {cond_severity_dist}")
    logger.info(f"  Condition body sites: {cond_body_site_dist}")
    logger.info(f"  Condition temporal relations: {cond_temporal_relation_dist}")
    logger.info(f"  Condition medication frequencies: {cond_frequency_dist}")
    logger.info(f"  Condition medication routes: {cond_route_dist}")
    logger.info(f"  Condition numeric ranges: {cond_range_count}")
    logger.info(f"  Condition value operators: {cond_value_operator_dist}")
    logger.info(f"  Recommendation strengths: {strength_dist}")
    return augmented_triples


# ──────────────────────────────────────────────────────────────────────
# Stage 4: Neo4j Knowledge Graph Construction
# ──────────────────────────────────────────────────────────────────────

def run_stage4(
    output_dir: str = None,
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None,
    neo4j_database: str = None,
    clear: bool = False,
    batch_size: int = None,
) -> dict:
    """Stage 4: load Stage 3 augmented triples and build a Neo4j KG.

    Lazy-imports neo4j_builder so stages 0-3 don't require the neo4j driver.
    """
    output_dir = output_dir or config.OUTPUT_DIR

    triples_path = os.path.join(output_dir, config.OUTPUT_AUGMENTED_TRIPLES_FILE)
    require_files(
        {"stage3_condition_augmented_triples": triples_path},
        hint="Run `python pipeline.py stage3` first.",
    )

    logger.info("=" * 60)
    logger.info("STAGE 4: Neo4j Knowledge Graph Construction")
    logger.info("=" * 60)

    from neo4j_builder import build_graph_from_file

    start_time = time.time()
    result = build_graph_from_file(
        triples_path=triples_path,
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
        clear_first=clear,
        batch_size=batch_size,
    )
    elapsed = time.time() - start_time

    summary_path = os.path.join(output_dir, config.OUTPUT_NEO4J_SUMMARY_FILE)
    save_json(
        {
            "metadata": {
                "stage": 4,
                "elapsed_seconds": round(elapsed, 1),
                "timestamp": datetime.now().isoformat(),
                "cleared_first": clear,
            },
            "result": result,
        },
        summary_path,
    )

    logger.info(f"Stage 4 complete in {elapsed:.1f}s")
    logger.info(f"  Output: {summary_path}")
    return result


# ──────────────────────────────────────────────────────────────────────
# Full Pipeline (all 5 stages)
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(
    db_name: list = None,
    xml_dir: str = None,
    primary_dir: str = None,
    pubmed_sqlite_path: str = None,
    semantic_groups_file: str = None,
    umls_api_key: str = None,
    openai_api_key: str = None,
    output_dir: str = None,
    max_guideline_text: int = None,
    max_guideline_docs: int = None,
    max_triples: int = None,
    batch_size: int = None,
    max_workers_umls: int = None,
    max_workers_llm: int = None,
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None,
    neo4j_database: str = None,
    neo4j_clear: bool = False,
    neo4j_batch_size: int = None,
    start_stage: int = 0,
    end_stage: int = 3,
):
    """Run the inclusive range [start_stage, end_stage] of pipeline stages."""
    output_dir = output_dir or config.OUTPUT_DIR
    db_name = db_name or ["PUBMED"]
    overall_start = time.time()

    logger.info("=" * 60)
    logger.info(f"PIPELINE: stages {start_stage}–{end_stage}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info("=" * 60)

    if start_stage <= 0 <= end_stage:
        run_stage0(
            db_name=db_name,
            output_dir=output_dir,
            max_guideline_text=max_guideline_text,
            xml_dir=xml_dir,
            primary_dir=primary_dir,
            pubmed_sqlite_path=pubmed_sqlite_path,
        )

    if start_stage <= 1 <= end_stage:
        run_stage1(
            output_dir=output_dir,
            openai_api_key=openai_api_key,
            max_workers=max_workers_llm,
            max_guideline_docs=max_guideline_docs,
        )

    if start_stage <= 2 <= end_stage:
        run_stage2(
            output_dir=output_dir,
            umls_api_key=umls_api_key,
            semantic_groups_file=semantic_groups_file,
            max_workers=max_workers_umls,
        )

    if start_stage <= 3 <= end_stage:
        run_stage3(
            output_dir=output_dir,
            openai_api_key=openai_api_key,
            batch_size=batch_size,
            max_triples=max_triples,
            max_workers=max_workers_llm,
        )

    if start_stage <= 4 <= end_stage:
        run_stage4(
            output_dir=output_dir,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            clear=neo4j_clear,
            batch_size=neo4j_batch_size,
        )

    elapsed = time.time() - overall_start
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE — total elapsed: {elapsed:.1f}s")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Medical KG Pipeline — unified entry for all stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(
        dest="stage", required=True,
        metavar="{stage0,stage1,stage2,stage3,stage4,all}",
    )

    # ── stage0 ──
    p0 = sub.add_parser("stage0", help="CREST parsing & recommendation extraction")
    p0.add_argument("--db-name", default=["CREST", "PUBMED"], nargs="+",
                    choices=["CREST", "PUBMED"],
                    help="One or more database sources for record/document IDs "
                         "(default: PUBMED). Pass multiple to combine, e.g. "
                         "--db-name CREST PUBMED")
    p0.add_argument("--xml-dir", default=None,
                    help=f"CREST xml/ folder (default: {config.CREST_XML_DIR})")
    p0.add_argument("--primary-dir", default=None,
                    help=f"CREST primary/ folder (default: {config.CREST_PRIMARY_DIR})")
    p0.add_argument("--pubmed-sqlite-path", default=None,
                    help=f"Path to PubMed SQLite (default: {config.PUBMED_SQLITE_PATH})")
    p0.add_argument("--output-dir", default=config.OUTPUT_DIR)
    p0.add_argument("--max-guideline-text", type=int, default=None,
                    help="Limit number of recommendations (test mode)")
    p0.add_argument("--log-level", default="INFO")

    # ── stage1 ──
    p1 = sub.add_parser("stage1", help="Entity candidate extraction (LLM)")
    p1.add_argument("--output-dir", default=config.OUTPUT_DIR)
    p1.add_argument("--openai-key", default=None)
    p1.add_argument("--max-workers", type=int, default=None,
                    help=f"Parallel LLM workers (default: {config.LLM_MAX_WORKERS})")
    p1.add_argument("--max-guideline-docs", type=int, default=10,
                    help="Limit documents to process at Stage 1 (test mode)")
    p1.add_argument("--log-level", default="INFO")

    # ── stage2 ──
    p2 = sub.add_parser("stage2", help="UMLS layer construction (triple extraction)")
    p2.add_argument("--output-dir", default=config.OUTPUT_DIR)
    p2.add_argument("--umls-key", default=None)
    p2.add_argument("--semantic-groups", default=None,
                    help=f"Path to semantic groups file "
                         f"(default: {config.SEMANTIC_GROUPS_FILE})")
    p2.add_argument("--max-workers", type=int, default=None,
                    help=f"Parallel UMLS workers (default: {config.UMLS_MAX_WORKERS})")
    p2.add_argument("--log-level", default="INFO")

    # ── stage3 ──
    p3 = sub.add_parser("stage3", help="Condition augmentation")
    p3.add_argument("--output-dir", default=config.OUTPUT_DIR)
    p3.add_argument("--openai-key", default=None)
    p3.add_argument("--batch-size", type=int, default=None,
                    help=f"Triples per LLM call (default: {config.STAGE3_LLM_CHUNK_SIZE})")
    p3.add_argument("--max-triples", type=int, default=None,
                    help="Limit triples to process (test mode)")
    p3.add_argument("--max-workers", type=int, default=None,
                    help=f"Parallel LLM workers (default: {config.LLM_MAX_WORKERS})")
    p3.add_argument("--log-level", default="INFO")

    # ── stage4 ──
    p4 = sub.add_parser("stage4", help="Neo4j knowledge-graph construction")
    p4.add_argument("--output-dir", default=config.OUTPUT_DIR)
    p4.add_argument("--neo4j-uri", default=None,
                    help=f"Neo4j Bolt URI (default: {config.NEO4J_URI})")
    p4.add_argument("--neo4j-user", default=None,
                    help=f"Neo4j username (default: {config.NEO4J_USER})")
    p4.add_argument("--neo4j-password", default=None,
                    help="Neo4j password (or set NEO4J_PASSWORD env var)")
    p4.add_argument("--neo4j-database", default=None,
                    help=f"Neo4j database (default: {config.NEO4J_DATABASE})")
    p4.add_argument("--clear", action="store_true",
                    help="DESTRUCTIVE: wipe the database before loading")
    p4.add_argument("--batch-size", type=int, default=None,
                    help=f"UNWIND rows per batch (default: {config.NEO4J_BATCH_SIZE})")
    p4.add_argument("--log-level", default="INFO")

    # ── all ──
    pall = sub.add_parser("all", help="End-to-end pipeline (Stages 0-4)")
    pall.add_argument("--db-name", default=["CREST"], nargs="+",
                      choices=["CREST", "PUBMED"],
                      help="One or more database sources passed through to Stage 0 "
                           "(default: PUBMED)")
    pall.add_argument("--xml-dir", default=None)
    pall.add_argument("--primary-dir", default=None)
    pall.add_argument("--pubmed-sqlite-path", default=None)
    pall.add_argument("--semantic-groups", default=None)
    pall.add_argument("--umls-key", default=None)
    pall.add_argument("--openai-key", default=None)
    pall.add_argument("--output-dir", default=config.OUTPUT_DIR)
    pall.add_argument("--max-guideline-text", type=int, default=None,
                      help="Limit recommendations/documents extracted at Stage 0")
    pall.add_argument("--max-guideline-docs", type=int, default=None,
                      help="Limit documents processed at Stage 1")
    pall.add_argument("--max-triples", type=int, default=None,
                      help="Limit triples at Stage 3")
    pall.add_argument("--batch-size", type=int, default=None,
                      help="Stage 3 LLM batch size")
    pall.add_argument("--max-workers-umls", type=int, default=None,
                      help=f"Stage 2 parallel workers "
                           f"(default: {config.UMLS_MAX_WORKERS})")
    pall.add_argument("--max-workers-llm", type=int, default=None,
                      help=f"Stage 1 / Stage 3 parallel LLM workers "
                           f"(default: {config.LLM_MAX_WORKERS})")
    pall.add_argument("--neo4j-uri", default=None)
    pall.add_argument("--neo4j-user", default=None)
    pall.add_argument("--neo4j-password", default=None)
    pall.add_argument("--neo4j-database", default=None)
    pall.add_argument("--neo4j-clear", action="store_true",
                      help="DESTRUCTIVE: wipe Neo4j database before Stage 4")
    pall.add_argument("--neo4j-batch-size", type=int, default=None)
    pall.add_argument("--start-stage", type=int, default=0,
                      choices=[0, 1, 2, 3, 4])
    pall.add_argument("--end-stage", type=int, default=3,
                      choices=[0, 1, 2, 3, 4],
                      help="Default 3; pass 4 to also build the Neo4j KG")
    pall.add_argument("--log-level", default="INFO")

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.stage == "stage0":
        run_stage0(
            db_name=args.db_name,
            output_dir=args.output_dir,
            max_guideline_text=args.max_guideline_text,
            xml_dir=args.xml_dir,
            primary_dir=args.primary_dir,
            pubmed_sqlite_path=args.pubmed_sqlite_path
        )
    elif args.stage == "stage1":
        run_stage1(
            output_dir=args.output_dir,
            openai_api_key=args.openai_key,
            max_workers=args.max_workers,
            max_guideline_docs=args.max_guideline_docs
        )
    elif args.stage == "stage2":
        run_stage2(
            output_dir=args.output_dir,
            umls_api_key=args.umls_key,
            semantic_groups_file=args.semantic_groups,
            max_workers=args.max_workers,
        )
    elif args.stage == "stage3":
        run_stage3(
            output_dir=args.output_dir,
            openai_api_key=args.openai_key,
            batch_size=args.batch_size,
            max_triples=args.max_triples,
            max_workers=args.max_workers,
        )
    elif args.stage == "stage4":
        run_stage4(
            output_dir=args.output_dir,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
            clear=args.clear,
            batch_size=args.batch_size,
        )
    elif args.stage == "all":
        if args.start_stage > args.end_stage:
            parser.error("--start-stage must be <= --end-stage")
        run_pipeline(
            db_name=args.db_name,
            xml_dir=args.xml_dir,
            primary_dir=args.primary_dir,
            pubmed_sqlite_path=args.pubmed_sqlite_path,
            semantic_groups_file=args.semantic_groups,
            umls_api_key=args.umls_key,
            openai_api_key=args.openai_key,
            output_dir=args.output_dir,
            max_guideline_text=args.max_guideline_text,
            max_guideline_docs=args.max_guideline_docs,
            max_triples=args.max_triples,
            batch_size=args.batch_size,
            max_workers_umls=args.max_workers_umls,
            max_workers_llm=args.max_workers_llm,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            neo4j_database=args.neo4j_database,
            neo4j_clear=args.neo4j_clear,
            neo4j_batch_size=args.neo4j_batch_size,
            start_stage=args.start_stage,
            end_stage=args.end_stage,
        )


if __name__ == "__main__":
    main()
