"""Analyze entity linking coverage for one extracted MediQ JSONL file.

Selects one input file under ``MediQ/extracted`` via ``input_type``, collects
unique extracted entities, and matches them with
``experiments.retrievers.CachedUMLSMatcher``.

Design points:
- reuse the same persistent UMLS cache across files
- avoid repeated matching within each file via ``unique_entity``
- stream input JSONL instead of loading entire files into memory
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from cli_utils import setup_logging

load_dotenv(PROJECT_ROOT / ".env", override=False)

from experiments.config import ExperimentConfig
from experiments.retrievers import CachedUMLSMatcher

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = os.path.join(".", "MediQ", "extracted")
DEFAULT_INPUT_TYPE = "mediQ_craft"
INPUT_TYPE_TO_FILE = {
    "MedQA": "medqa_dev_convo.jsonl",
    "mediQ_craft": "all_craft_md.jsonl",
    "mediQ_dev": "all_dev_good.jsonl",
}


def _build_cfg(args: argparse.Namespace, dataset_path: str) -> ExperimentConfig:
    cfg = ExperimentConfig(
        dataset_name="mediq_extracted_analysis",
        dataset_path=dataset_path,
        output_dir=args.input_dir,
        max_samples=args.max_samples,
        start_index=args.start_index,
    )
    cfg.cache_dir = args.input_dir
    if args.log_level:
        cfg.log_level = args.log_level
    os.makedirs(cfg.cache_dir, exist_ok=True)
    logger.info(
        "Configured analysis: dataset_path=%s, cache_dir=%s, max_samples=%s, start_index=%d",
        dataset_path,
        cfg.cache_dir,
        str(args.max_samples),
        args.start_index,
    )
    return cfg


def _build_matcher_cfg(cfg: ExperimentConfig, args: argparse.Namespace) -> SimpleNamespace:
    cache_path = os.path.join(args.input_dir, f"{args.input_type}_umls_match_cache.json")
    logger.info("Using input-type-specific UMLS cache path: %s", cache_path)
    return SimpleNamespace(
        umls_api_key=cfg.umls_api_key,
        semantic_groups_file=cfg.semantic_groups_file,
        umls_match_cache_path=cache_path,
    )


def _iter_jsonl(path: str, max_samples: int | None, start_index: int):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    yielded = 0
    seen = 0
    skipped_invalid = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                skipped_invalid += 1
                logger.warning(
                    "Skipping malformed JSONL record in %s at line %d: %s | snippet=%r",
                    path,
                    line_no,
                    exc,
                    raw_line[:200],
                )
                continue

            if seen < start_index:
                seen += 1
                continue
            if max_samples is not None and yielded >= max_samples:
                break

            seen += 1
            yielded += 1
            yield record

    if skipped_invalid:
        logger.warning(
            "Finished reading %s with %d malformed JSONL line(s) skipped",
            path,
            skipped_invalid,
        )


def _iter_entities(record: dict):
    extracted_info = record.get("extracted_info") or {}
    if not isinstance(extracted_info, dict):
        return

    sample_infos = extracted_info.get("info")
    if isinstance(sample_infos, dict):
        iterable = sample_infos.values()
    else:
        iterable = extracted_info.values()

    for sample_info in iterable:
        if not isinstance(sample_info, dict):
            continue
        entities = sample_info.get("entities") or []
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if isinstance(entity, dict):
                yield entity


def _process_file(
    input_path: str,
    input_type: str,
    matcher: CachedUMLSMatcher,
    max_samples: int | None,
    start_index: int,
) -> dict:
    logger.info("Starting analysis for input_type=%s, input_path=%s", input_type, input_path)
    sample_count = 0
    entity_mentions = 0
    unique_entity: dict[str, dict] = {}

    for record in _iter_jsonl(input_path, max_samples=max_samples, start_index=start_index):
        sample_count += 1
        for entity in _iter_entities(record):
            entity_mentions += 1
            key = CachedUMLSMatcher._key(entity)
            if not key or key in unique_entity:
                continue
            unique_entity[key] = entity
        if sample_count % 100 == 0:
            logger.info(
                "Scanned %d samples from %s: entity_mentions=%d, unique_entities=%d",
                sample_count,
                os.path.basename(input_path),
                entity_mentions,
                len(unique_entity),
            )

    unique_entities = list(unique_entity.values())
    logger.info(
        "Completed entity scan for %s: samples=%d, entity_mentions=%d, unique_entities=%d",
        os.path.basename(input_path),
        sample_count,
        entity_mentions,
        len(unique_entities),
    )

    logger.info(
        "Starting UMLS matching for %s with %d unique entities",
        os.path.basename(input_path),
        len(unique_entities),
    )
    match_results, cuis = matcher.match_many(unique_entities)
    logger.info(
        "Completed UMLS matching for %s: match_results=%d, unique_cuis=%d",
        os.path.basename(input_path),
        len(match_results),
        len(cuis),
    )

    matched_entities = 0
    unmatched_entities = 0
    match_types: dict[str, int] = {}

    for result in match_results:
        if result.get("matched"):
            matched_entities += 1
        else:
            unmatched_entities += 1
        match_type = str(result.get("match_type") or "unknown")
        match_types[match_type] = match_types.get(match_type, 0) + 1

    summary = {
        "input_type": input_type,
        "file_name": os.path.basename(input_path),
        "file_path": input_path,
        "sample_count": sample_count,
        "entity_mentions": entity_mentions,
        "unique_entity_count": len(unique_entities),
        "matched_entity_count": matched_entities,
        "unmatched_entity_count": unmatched_entities,
        "unique_cui_count": len(cuis),
        "match_types": match_types,
        "match_results": match_results,
    }
    logger.info(
        "%s: samples=%d, mentions=%d, unique_entities=%d, matched=%d, cuis=%d",
        summary["file_name"],
        sample_count,
        entity_mentions,
        summary["unique_entity_count"],
        matched_entities,
        len(cuis),
    )
    return summary


def _resolve_input_path(args: argparse.Namespace) -> str:
    filename = INPUT_TYPE_TO_FILE[args.input_type]
    input_path = os.path.join(args.input_dir, filename)
    logger.info(
        "Resolved input_type=%s to input_path=%s",
        args.input_type,
        input_path,
    )
    return input_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze extracted MediQ entity files with CachedUMLSMatcher."
        ),
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--input-type",
        default=DEFAULT_INPUT_TYPE,
        choices=tuple(INPUT_TYPE_TO_FILE.keys()),
        help="Choose which extracted JSONL file to analyze",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    logger.info("Loaded environment and initialized logging at level=%s", args.log_level)

    input_path = _resolve_input_path(args)
    output_file = f"{args.input_type}_umls_match.json"
    summary_path = os.path.join(args.input_dir, output_file)
    logger.info("Output summary will be written to %s", summary_path)

    cfg = _build_cfg(args, dataset_path=input_path)
    matcher_cfg = _build_matcher_cfg(cfg, args)
    logger.info("Initializing CachedUMLSMatcher with cache=%s", matcher_cfg.umls_match_cache_path)
    matcher = CachedUMLSMatcher(matcher_cfg)

    try:
        summary = _process_file(
            input_path=input_path,
            input_type=args.input_type,
            matcher=matcher,
            max_samples=args.max_samples,
            start_index=args.start_index,
        )
    finally:
        logger.info("Flushing UMLS matcher cache to %s", matcher_cfg.umls_match_cache_path)
        matcher.flush()

    payload = {
        "input_type": args.input_type,
        "input_dir": args.input_dir,
        "input_path": input_path,
        "umls_match_cache_path": matcher_cfg.umls_match_cache_path,
        "output_file": output_file,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    logger.info("Writing analysis payload to %s", summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
