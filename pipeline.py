"""
Medical KG Pipeline — Main Orchestration
Stage 1: Entity Candidate Extraction (CREST + LLM)
Stage 2: UMLS Layer Construction (UMLS REST API)

Usage:
    python pipeline.py --umls-key YOUR_KEY --openai-key YOUR_KEY
    python pipeline.py --xml-dir ./crest/xml --primary-dir ./crest/primary
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import config
from crest_parser import extract_from_both_sources
from entity_extractor import (
    call_openai,
    extract_entities_batch,
    deduplicate_entities,
)
from semantic_types import load_semantic_groups_from_file
from UMLS_KG.umls_client import UMLSClient
from entity_matcher import match_entities_batch
from search_KG import search_KG
from subgraph_builder import build_subgraphs_batch, deduplicate_triples

logger = logging.getLogger(__name__)
MEDIQ_QUERY_FILES = (
    "all_dev_good.jsonl",
    "all_craft_md.jsonl",
)


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def save_json(data, filepath: str):
    """Save data as JSON with UTF-8 encoding."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Saved: {filepath}")


def load_jsonl(filepath: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Skipping invalid JSONL row in {filepath}:{line_no}: {e}"
                )
    return records


def run_stage1(
    recommendations: list[dict],
    llm_fn=None,
    progress_interval: int = 10,
) -> tuple[list[dict], dict]:
    """
    Stage 1: Entity Candidate Extraction

    Input: CREST recommendation sentences (with guideline context)
    Output: deduplicated entity candidates with source metadata
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Entity Candidate Extraction")
    logger.info("=" * 60)

    if llm_fn is None:
        llm_fn = call_openai

    all_entities = extract_entities_batch(
        recommendations, llm_fn=llm_fn, progress_interval=progress_interval,
    )

    unique_entities = deduplicate_entities(all_entities)
    return all_entities, unique_entities


def run_stage2(
    unique_entities: dict,
    umls_client: UMLSClient,
    tui_to_group: dict,
    progress_interval: int = 50,
) -> tuple[list[dict], dict, list[dict]]:
    """
    Stage 2: UMLS Layer Construction

    Input: deduplicated entity candidates
    Output: match results, matched CUIs, 1-hop triples
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: UMLS Layer Construction")
    logger.info("=" * 60)

    logger.info("Stage 2-A: Entity Candidate → UMLS CUI matching")
    match_results, matched_cuis = match_entities_batch(
        unique_entities, umls_client, tui_to_group,
        progress_interval=progress_interval,
    )

    logger.info("Stage 2-B: 1-hop subgraph collection")
    all_triples = build_subgraphs_batch(
        umls_client, matched_cuis, progress_interval=progress_interval,
    )

    unique_triples = deduplicate_triples(all_triples)
    return match_results, matched_cuis, unique_triples


def get_query_entity(
    mediq_dir: str = "mediQ_ER",
    output_dir: str = None,
    llm_fn=None,
    progress_interval: int = 10,
):
    """
    Extract entities from each `context` sentence in the mediQ_ER JSONL files.

    Each item in a record's `context` list is converted into a Stage 1-style
    recommendation input and processed sequentially through entity_extractor.py.
    """
    output_dir = output_dir or config.OUTPUT_DIR
    query_output_dir = os.path.join(output_dir, "mediQ_ER")
    os.makedirs(query_output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("QUERY ENTITY EXTRACTION: mediQ_ER")
    logger.info("=" * 60)

    file_summaries = []
    combined_all_entities = []
    combined_records = []

    for filename in MEDIQ_QUERY_FILES:
        input_path = os.path.join(mediq_dir, filename)
        if not os.path.exists(input_path):
            logger.warning(f"Skipping missing mediQ_ER file: {input_path}")
            continue

        records = load_jsonl(input_path)
        recommendations = []

        for record in records:
            record_id = record.get("id", "")
            question = record.get("question", "")
            contexts = record.get("context", [])

            if not isinstance(contexts, list):
                logger.warning(
                    f"Skipping non-list context in {filename} record {record_id}"
                )
                continue

            guideline_context = question.strip()
            for idx, context_text in enumerate(contexts):
                if not isinstance(context_text, str) or not context_text.strip():
                    continue

                source_id = f"{filename}::{record_id}::{idx}"
                recommendations.append(
                    {
                        "guideline_id": source_id,
                        "strength": "",
                        "text": context_text.strip(),
                        "guideline_context": guideline_context,
                    }
                )
        logger.info(
            f"{filename}: loaded {len(records)} records, "
            f"{len(recommendations)} context sentences"
        )

        all_entities = extract_entities_batch(
            recommendations,
            llm_fn=llm_fn,
            progress_interval=progress_interval,
        )
        unique_entities = deduplicate_entities(all_entities)
        combined_all_entities.extend(all_entities)

        entities_by_source = {}
        for ent in all_entities:
            source_id = ent.get("source_guideline_id", "")
            entities_by_source.setdefault(source_id, []).append(ent)

        input_query_results = []
        for record in records:
            record_id = record.get("id", "")
            contexts = record.get("context", [])
            input_query_entity = []

            if not isinstance(contexts, list):
                continue

            for idx, context_text in enumerate(contexts):
                source_id = f"{filename}::{record_id}::{idx}"
                input_query_entity.append(
                    {
                        "context_index": idx,
                        "text": context_text,
                        "entities": entities_by_source.get(source_id, []),
                    }
                )

            input_query_results.append(
                {
                    "source_file": filename,
                    "id": record_id,
                    "question": record.get("question", ""),
                    "answer": record.get("answer", ""),
                    "answer_idx": record.get("answer_idx", ""),
                    "input_query_entity": input_query_entity,
                }
            )
        combined_records.extend(input_query_results)

        output_path = os.path.join(
            query_output_dir,
            filename.replace(".jsonl", "_query_entities.json"),
        )
        save_json(
            {
                "source_file": input_path,
                "total_records": len(records),
                "total_context_sentences": len(recommendations),
                "total_raw_entities": len(all_entities),
                "total_unique_entities": len(unique_entities),
                "records": input_query_results,
                "unique_entities": list(unique_entities.values()),
            },
            output_path,
        )

        file_summaries.append(
            {
                "source_file": input_path,
                "output_file": output_path,
                "records": len(records),
                "context_sentences": len(recommendations),
                "raw_entities": len(all_entities),
                "unique_entities": len(unique_entities),
            }
        )

    combined_unique_entities = deduplicate_entities(combined_all_entities)
    input_data_summary = {
        "files": file_summaries,
        "total_files": len(file_summaries),
        "total_raw_entities": len(combined_all_entities),
        "total_unique_entities": len(combined_unique_entities),
        "records": combined_records,
        "unique_entities": list(combined_unique_entities.values()),
    }
    summary_path = os.path.join(query_output_dir, "query_entity_summary.json")
    save_json(input_data_summary, summary_path)
    return input_data_summary


def run_pipeline(
    xml_dir: str = None,
    primary_dir: str = None,
    semantic_types_file: str = None,
    output_dir: str = None,
    max_recommendations: int = None,
):
    """Run the full Stage 1 + Stage 2 pipeline."""
    start_time = time.time()

    xml_dir = xml_dir or config.CREST_XML_DIR
    primary_dir = primary_dir or config.CREST_PRIMARY_DIR
    semantic_types_file = semantic_types_file or config.SEMANTIC_TYPES_FILE
    output_dir = output_dir or config.OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Medical KG Pipeline — Stage 1 & Stage 2")
    logger.info(f"  CREST xml/: {xml_dir}")
    logger.info(f"  CREST primary/: {primary_dir}")
    logger.info(f"  Output: {output_dir}")

    # ── Load Semantic Type Mappings ──
    tui_to_group, tui_to_name = load_semantic_groups_from_file(semantic_types_file)
    logger.info(f"Loaded {len(tui_to_group)} semantic type → group mappings")

    # ── Parse CREST Corpus ──
    recommendations = extract_from_both_sources(
        xml_dir=xml_dir, primary_dir=primary_dir,
    )

    if not recommendations:
        logger.error("No recommendations extracted. Check CREST paths.")
        return

    if max_recommendations:
        recommendations = recommendations[:max_recommendations]
        logger.info(f"Limited to {max_recommendations} recommendations (test mode)")

    # ── Save recommendations for Stage 3 reuse ──
    save_json(
        {
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
        },
        os.path.join(output_dir, config.OUTPUT_RECOMMENDATIONS_FILE),
    )

    # ── Stage 1 ──
    all_entities, unique_entities = run_stage1(recommendations)

    save_json(
        {
            "total_raw_entities": len(all_entities),
            "total_unique_entities": len(unique_entities),
            "entities": list(unique_entities.values()),
        },
        os.path.join(output_dir, config.OUTPUT_ENTITIES_FILE),
    )

    # ── Stage 2 ──
    umls_client = UMLSClient()
    match_results, matched_cuis, unique_triples = run_stage2(
        unique_entities, umls_client, tui_to_group,
    )

    save_json(
        {
            "total_match_results": len(match_results),
            "matched_count": sum(1 for r in match_results if r["matched"]),
            "total_unique_cuis": len(matched_cuis),
            "match_results": match_results,
            "matched_cuis": matched_cuis,
        },
        os.path.join(output_dir, config.OUTPUT_MATCHED_FILE),
    )

    save_json(
        {
            "total_triples": len(unique_triples),
            "triples": unique_triples,
        },
        os.path.join(output_dir, config.OUTPUT_TRIPLES_FILE),
    )

    # ── Pipeline Summary ──
    elapsed = time.time() - start_time
    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "crest_recommendations": len(recommendations),
        "stage1_raw_entities": len(all_entities),
        "stage1_unique_entities": len(unique_entities),
        "stage2_matched_entities": sum(1 for r in match_results if r["matched"]),
        "stage2_match_rate": round(
            sum(1 for r in match_results if r["matched"])
            / max(len(match_results), 1) * 100, 1,
        ),
        "stage2_unique_cuis": len(matched_cuis),
        "stage2_total_triples": len(unique_triples),
        "umls_api_requests": umls_client.request_count,
        "triples": unique_triples,
    }

    save_json(
        {k: v for k, v in summary.items() if k != "triples"},
        os.path.join(output_dir, config.OUTPUT_PIPELINE_LOG),
    )

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Time elapsed: {elapsed:.1f}s")
    logger.info(f"  Recommendations processed: {summary['crest_recommendations']}")
    logger.info(f"  Entity candidates (unique): {summary['stage1_unique_entities']}")
    logger.info(f"  UMLS match rate: {summary['stage2_match_rate']}%")
    logger.info(f"  Unique CUIs: {summary['stage2_unique_cuis']}")
    logger.info(f"  UMLS Layer triples: {summary['stage2_total_triples']}")
    logger.info(f"  UMLS API requests: {summary['umls_api_requests']}")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Medical KG Pipeline: Stage 1 (Entity Extraction) + Stage 2 (UMLS Layer)"
    )
    parser.add_argument("--xml-dir", type=str, help="Path to CREST xml/ folder")
    parser.add_argument("--primary-dir", type=str, help="Path to CREST primary/ folder")
    parser.add_argument(
        "--semantic-types",
        type=str,
        help="Path to semantic_type_of_UMLS.json",
    )
    parser.add_argument(
        "--semantic-groups",
        dest="semantic_types",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--umls-key", type=str, help="UMLS REST API key")
    parser.add_argument("--openai-key", type=str, help="OpenAI API key")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--max-recs", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()
    setup_logging(args.log_level)

    pipeline_summary = run_pipeline(
        xml_dir=args.xml_dir,
        primary_dir=args.primary_dir,
        semantic_types_file=args.semantic_types,
        output_dir=args.output_dir,
        max_recommendations=args.max_recs,
    )

    if pipeline_summary is not None:
        input_data_summary = get_query_entity(output_dir=args.output_dir)
        search_result = search_KG(
            kg_info=pipeline_summary,
            input_info=input_data_summary,
        )
        save_json(
            search_result,
            os.path.join(args.output_dir, "mediQ_ER", "search_kg_result.json"),
        )


if __name__ == "__main__":
    main()
