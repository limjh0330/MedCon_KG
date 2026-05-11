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
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

import condition_augmenter
import config

from cli_utils import load_json, require_files, save_json, setup_logging
from crest_parser import extract_from_both_sources
from dataset.Pubmed.get_pubmed_data import extract_from_sqlite

from entity_extractor import deduplicate_entities, extract_entities_batch
from entity_matcher import match_entities_batch
from semantic_types import load_semantic_groups_from_file
from subgraph_builder import build_subgraphs_batch, deduplicate_triples
from umls_client import UMLSClient

logger = logging.getLogger(__name__)


def _build_stage0_outputs(
    db_name: str,
    max_guideline_text: int = None,
) -> tuple[list[dict], list[dict], dict[str, dict[str, Any]]]:
    """Normalize Stage 0 rows into DB/document-centric sentence records."""
    records: list[dict] = []
    documents_by_id: dict[str, dict[str, Any]] = {}
    
    guideline_text = None
    if "CREST" in db_name:
        guideline_text = extract_from_both_sources(
            xml_dir=config.CREST_XML_DIR,
            primary_dir=config.CREST_PRIMARY_DIR,
        )

        if not guideline_text:
            logger.error("No recommendations extracted. Check CREST paths.")
            sys.exit(1)

    elif "PUBMED" in db_name: 
        pubmed_guideline_text = extract_from_sqlite(config.PUBMED_SQLITE_PATH)
        
        if guideline_text is None:
            guideline_text = pubmed_guideline_text
        else:
            guideline_text.extend(pubmed_guideline_text)

    if max_guideline_text:
        guideline_text = guideline_text[:max_guideline_text]
        logger.info(f"Limited to {max_guideline_text} recommendations (test mode)")

    for sentence_index, rec in enumerate(guideline_text, start=1):
        guideline_id = str(rec.get("guideline_id", "")).strip() or f"doc_{sentence_index}"
        db_guideline_id = f"{db_name}_{guideline_id}"
        raw_text = (rec.get("guideline_context") or "").strip()

        normalized = dict(rec)
        normalized["db_guideline_id"] = db_guideline_id
        normalized["raw_text"] = raw_text
        records.append(normalized)

        if db_guideline_id not in documents_by_id:
            documents_by_id[db_guideline_id] = {
                "db_guideline_id": db_guideline_id,
                "guideline_context": rec.get("guideline_context", ""),
                "raw_texts": [],
                "sentence_count": 0,
            }

        doc_entry = documents_by_id[db_guideline_id]
        doc_entry["raw_texts"].append(raw_text)
        doc_entry["sentence_count"] += 1

    documents = list(documents_by_id.values())
    return records, documents, documents_by_id


# ──────────────────────────────────────────────────────────────────────
# Stage 0: CREST Parsing & Recommendation/Context Extraction
# ──────────────────────────────────────────────────────────────────────

def run_stage0(
    db_name: list = None, 
    output_dir: str = None,
    max_guideline_text: int = None,
) -> list[dict]:
    """Stage 0: parse CREST → save recommendations file. Returns the list."""
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STAGE 0: Guideline parsing & extraction")
    logger.info("=" * 60)

    start = time.time()

    records, documents, documents_by_id = _build_stage0_outputs(db_name, max_guideline_text)

    elapsed = time.time() - start
    output_path = os.path.join(output_dir, config.OUTPUT_RECOMMENDATIONS_FILE)

    save_json(
        {
            "metadata": {
                "stage": 0,
                "db_name": db_name,
                "total_recommendations": len(records),
                "total_documents": len(documents),
                "elapsed_seconds": round(elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            },
            "recommendations": records,
            "records": records,
            "documents": documents,
            "documents_by_id": documents_by_id,
        },
        output_path,
    )

    logger.info(f"Stage 0 complete in {elapsed:.1f}s")
    logger.info(f"  DB name: {db_name}")
    logger.info(f"  Raw-text records: {len(records)}")
    logger.info(f"  Document IDs: {len(documents)}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"Next: python pipeline.py stage1 --output-dir {output_dir}")
    return records


# ──────────────────────────────────────────────────────────────────────
# Stage 1: Entity Candidate Extraction (LLM)
# ──────────────────────────────────────────────────────────────────────

def run_stage1(
    output_dir: str = None,
    openai_api_key: str = None,
    max_workers: int = None,
    max_guideline_text: int = None,
    progress_interval: int = 10,
) -> tuple[list[dict], dict]:
    """Stage 1: load Stage 0 records, run LLM extraction, save entities."""
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
    records = recs_data.get("records") or recs_data.get("recommendations", [])
    documents = recs_data.get("documents", [])
    logger.info(f"  Loaded {len(records)} raw-text records from Stage 0")
    if documents:
        logger.info(f"  Loaded {len(documents)} document IDs from Stage 0")

    if max_guideline_text:
        records = records[:max_guideline_text]
        logger.info(f"  Limited to {max_guideline_text} raw-text records (test mode)")

    start = time.time()
    all_entities = extract_entities_batch(
        records,
        progress_interval=progress_interval,
        max_workers=max_workers,
    )
    unique_entities = deduplicate_entities(all_entities)
    elapsed = time.time() - start

    output_path = os.path.join(output_dir, config.OUTPUT_ENTITIES_FILE)
    save_json(
        {
            "metadata": {
                "stage": 1,
                "input_records": len(records),
                "input_documents": len(documents),
                "total_raw_entities": len(all_entities),
                "total_unique_entities": len(unique_entities),
                "elapsed_seconds": round(elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            },
            "entities": list(unique_entities.values()),
        },
        output_path,
    )

    logger.info(f"Stage 1 complete in {elapsed:.1f}s")
    logger.info(f"  Raw entities: {len(all_entities)}")
    logger.info(f"  Unique entities: {len(unique_entities)}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"Next: python pipeline.py stage2 --output-dir {output_dir}")
    return all_entities, unique_entities


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
    for t in augmented_triples:
        for c in t.get("conditions", []):
            ct = c.get("type", "unknown")
            cond_type_dist[ct] = cond_type_dist.get(ct, 0) + 1

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
    pubmed_sqlite_path: str = None,
    semantic_groups_file: str = None,
    umls_api_key: str = None,
    openai_api_key: str = None,
    output_dir: str = None,
    max_guideline_text: int = None,
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
        )

    if start_stage <= 1 <= end_stage:
        run_stage1(
            output_dir=output_dir,
            openai_api_key=openai_api_key,
            max_workers=max_workers_llm,
            max_guideline_text=max_guideline_text,
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
    p0.add_argument("--db-name", default=["PUBMED"], 
                    help="Database name for record/document IDs "
                         "(default: 'CREST', 'PUBMED')")
    p0.add_argument("--xml-dir", default=None,
                    help=f"CREST xml/ folder (default: {config.CREST_XML_DIR})")
    p0.add_argument("--primary-dir", default=None,
                    help=f"CREST primary/ folder (default: {config.CREST_PRIMARY_DIR})")
    p0.add_argument("--pubmed-sqlite-path", default=None,
                    help=f"Path to PubMed SQLite (default: {config.PUBMED_SQLITE_PATH})")
    p0.add_argument("--output-dir", default=config.OUTPUT_DIR)
    p0.add_argument("--max-guideline-text", type=int, default=10,
                    help="Limit number of recommendations (test mode)")
    p0.add_argument("--log-level", default="INFO")

    # ── stage1 ──
    p1 = sub.add_parser("stage1", help="Entity candidate extraction (LLM)")
    p1.add_argument("--output-dir", default=config.OUTPUT_DIR)
    p1.add_argument("--openai-key", default=None)
    p1.add_argument("--max-workers", type=int, default=None,
                    help=f"Parallel LLM workers (default: {config.LLM_MAX_WORKERS})")
    p1.add_argument("--max-guideline-text", type=int, default=None,
                    help="Limit recommendations to process (test mode)")
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
    pall.add_argument("--xml-dir", default=None)
    pall.add_argument("--primary-dir", default=None)
    pall.add_argument("--pubmed-sqlite-path", default=None)
    pall.add_argument("--semantic-groups", default=None)
    pall.add_argument("--umls-key", default=None)
    pall.add_argument("--openai-key", default=None)
    pall.add_argument("--output-dir", default=config.OUTPUT_DIR)
    pall.add_argument("--max-guideline-text", type=int, default=None,
                      help="Limit recommendations at Stage 0/1")
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
        )
    elif args.stage == "stage1":
        run_stage1(
            output_dir=args.output_dir,
            openai_api_key=args.openai_key,
            max_workers=args.max_workers,
            max_guideline_text=args.max_guideline_text,
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
            pubmed_sqlite_path=args.pubmed_sqlite_path,
            semantic_groups_file=args.semantic_groups,
            umls_api_key=args.umls_key,
            openai_api_key=args.openai_key,
            output_dir=args.output_dir,
            max_guideline_text=args.max_guideline_text,
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
