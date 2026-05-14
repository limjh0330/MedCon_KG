"""Extract entities and conditions for MediQ jsonl files.

Uses the same sentence splitting, entity extraction, and condition extraction
flow as `KGWithCondRunner`, but stops after writing structured extraction
results back to jsonl.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli_utils import setup_logging

from experiments.config import ExperimentConfig
from experiments.datasets import Sample
from experiments.llm_backend import LocalLLM
from experiments.pipeline import (
    LlamaConditionExtractor,
    LlamaEntityExtractor,
    conditions_to_keywords,
    split_sentences,
)

logger = logging.getLogger(__name__)

DEFAULT_MEDIQ_DIR = "./MediQ"
DEFAULT_INPUT_FILES = (
    "all_craft_md.jsonl",
    "all_dev_good.jsonl",
    "medqa_dev_convo.jsonl",
)
_JSON_DUMP_KWARGS = {"ensure_ascii": False, "separators": (",", ":")}


def _build_cfg(args: argparse.Namespace, dataset_path: str, output_dir: str) -> ExperimentConfig:
    cfg = ExperimentConfig(
        dataset_name="mediq",
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_samples=args.max_samples,
        start_index=args.start_index,
    )
    cfg.variants = ("kg_with_cond",)
    if args.llm_model:
        cfg.llm_model_name = args.llm_model
    if args.llm_dtype:
        cfg.llm_dtype = args.llm_dtype
    if args.llm_attn_impl:
        cfg.llm_attn_impl = args.llm_attn_impl
    if args.llm_batch_size is not None:
        cfg.llm_batch_size = args.llm_batch_size
    if args.condition_sentence_chunk_size is not None:
        cfg.llm_condition_sentence_chunk_size = max(1, args.condition_sentence_chunk_size)
    if args.log_level:
        cfg.log_level = args.log_level
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    return cfg


def _extract_sample_info(
    sample,
    entity_extractor: LlamaEntityExtractor,
    condition_extractor: LlamaConditionExtractor,
    cfg: ExperimentConfig,
) -> dict:
    sentences = split_sentences(sample.input_query())
    per_sent_entities = entity_extractor.extract_per_sentence(sentences)
    seen_entities: set[str] = set()
    unique_entities: list[dict] = []
    sentence_entities = [
        {
            "sentence_index": idx,
            "sentence": sentence,
            "entities": ents,
        }
        for idx, (sentence, ents) in enumerate(zip(sentences, per_sent_entities))
        if ents
    ]
    total_entity_count = 0
    for ents in per_sent_entities:
        total_entity_count += len(ents)
        for ent in ents:
            key = (
                str(ent.get("normalized_form") or ent.get("surface_form") or "")
                .lower()
                .strip()
            )
            if not key or key in seen_entities:
                continue
            seen_entities.add(key)
            unique_entities.append(ent)

    per_sent_conditions = condition_extractor.extract_per_sentence(sentences)
    all_conditions: list[dict] = []
    unique_conditions: set[str] = set()
    for conds in per_sent_conditions:
        for cond in conds:
            if not isinstance(cond, dict):
                continue
            all_conditions.append(cond)
            unique_conditions.add(
                json.dumps(cond, ensure_ascii=False, sort_keys=True)
            )

    condition_keywords = conditions_to_keywords(
        all_conditions,
        min_len=cfg.cond_keyword_min_len,
        max_count=cfg.cond_keyword_max,
    )

    return {
        "meta_data": {
            "sentence_count": len(sentences),
            "entity_sentence_count": len(sentence_entities),
            "total_entity_count": total_entity_count,
            "unique_entity_count": len(unique_entities),
            "total_condition_count": len(all_conditions),
            "unique_condition_count": len(unique_conditions),
        },
        "info": {
            str(sample.sample_id): {
                "sentence_entities": sentence_entities,
                "entities": unique_entities,
                "conditions": all_conditions,
                "condition_keywords": condition_keywords,
            }
        },
    }


def _iter_mediq_samples(
    path: str,
    max_samples: int | None,
    start_index: int,
):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    yielded = 0
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if seen < start_index:
                seen += 1
                continue
            if max_samples is not None and yielded >= max_samples:
                break

            r = json.loads(line)
            seen += 1
            yielded += 1
            yield Sample(
                sample_id=str(r.get("id", "")),
                question=r.get("question", "") or "",
                context=r.get("context", []) or [],
                options=r.get("options", {}) or {},
                gold_answer=r.get("answer", "") or "",
                gold_answer_idx=r.get("answer_idx", "") or "",
                raw=r,
            )


def _process_file(
    input_path: str,
    output_path: str,
    llm: LocalLLM,
    args: argparse.Namespace,
) -> None:
    cfg = _build_cfg(args, dataset_path=input_path, output_dir=args.output_dir)
    entity_extractor = LlamaEntityExtractor(llm, cfg)
    condition_extractor = LlamaConditionExtractor(llm, cfg)

    logger.info("Processing %s -> %s", input_path, output_path)
    count = 0
    with open(output_path, "w", encoding="utf-8", buffering=1024 * 1024) as out_f:
        for sample in _iter_mediq_samples(
            path=input_path,
            max_samples=args.max_samples,
            start_index=args.start_index,
        ):
            sample.raw["extracted_info"] = _extract_sample_info(
                sample,
                entity_extractor=entity_extractor,
                condition_extractor=condition_extractor,
                cfg=cfg,
            )
            out_f.write(json.dumps(sample.raw, **_JSON_DUMP_KWARGS))
            out_f.write("\n")
            count += 1
            if count % 25 == 0:
                out_f.flush()
                logger.info("  processed %d samples from %s", count, os.path.basename(input_path))
    logger.info("Completed %s (%d samples)", input_path, count)


def _resolve_input_files(args: argparse.Namespace) -> list[str]:
    if args.input_files:
        return [os.path.join(args.input_dir, name) for name in args.input_files]
    return [os.path.join(args.input_dir, name) for name in DEFAULT_INPUT_FILES]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract entities and conditions for the three MediQ jsonl files.",
    )
    parser.add_argument("--input-dir", default=DEFAULT_MEDIQ_DIR)
    parser.add_argument("--input-files", nargs="+", default=None,
                        help="Optional subset of MediQ jsonl filenames to process")
    parser.add_argument("--output-dir", default=os.path.join(DEFAULT_MEDIQ_DIR, "extracted"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-dtype", default=None, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--llm-attn-impl", default=None, choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--llm-batch-size", type=int, default=None)
    parser.add_argument("--condition-sentence-chunk-size", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    os.makedirs(args.output_dir, exist_ok=True)
    cfg_for_model = _build_cfg(
        args,
        dataset_path=os.path.join(args.input_dir, DEFAULT_INPUT_FILES[0]),
        output_dir=args.output_dir,
    )

    logger.info("Loading local model once for all input files")
    llm = LocalLLM(
        model_name=cfg_for_model.llm_model_name,
        dtype=cfg_for_model.llm_dtype,
        attn_impl=cfg_for_model.llm_attn_impl,
        device=cfg_for_model.llm_device,
        hf_token=cfg_for_model.llm_hf_token,
        deterministic=cfg_for_model.deterministic,
    )

    for input_path in _resolve_input_files(args):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        output_path = os.path.join(args.output_dir, os.path.basename(input_path))
        _process_file(
            input_path=input_path,
            output_path=output_path,
            llm=llm,
            args=args,
        )


if __name__ == "__main__":
    main()
