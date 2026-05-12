"""End-to-end RAG experiment driver.

Usage (single GPU H100 80GB, Ubuntu 22.04, CUDA 12.4, PyTorch 2.4, Python 3.11):

  # ── one-time setup ──
  pip install -r experiments/requirements.txt
  export HF_TOKEN=hf_xxx          # needed for meta-llama/Llama-3.1-8B-Instruct
  export OPENAI_API_KEY=sk-xxx    # for text-embedding-3-large (Variant 2)
  export UMLS_API_KEY=xxx         # for entity → CUI matching (Variants 3, 4)

  # ── full MediQ, all four variants ──
  python -m experiments.run \\
      --dataset mediq \\
      --dataset-path ./MediQ/all_craft_md.jsonl \\
      --variants all \\
      --output-dir ./output/full_rag_experiment

  # ── quick smoke test (first 5 samples, all 4 variants) ──
  python -m experiments.run --max-samples 5 --variants all

  # ── only one variant ──
  python -m experiments.run --variants vector_rag,kg_with_cond

Outputs in --output-dir:
  results.json   : per-sample × per-variant records (predicted, gold, correct, elapsed_seconds, retrieval debug)
  summary.json   : aggregate accuracy + total/mean elapsed per variant + config snapshot
  trace.log      : human-readable per-sample trace
  cache/         : OpenAI embeddings + UMLS match results (persistent across runs)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Iterable, Optional, TextIO

from cli_utils import setup_logging

from experiments.config import ALL_VARIANTS, ExperimentConfig
from experiments.datasets import Sample, load_dataset
from experiments.llm_backend import LocalLLM
from experiments.pipeline import (
    KGNoCondRunner,
    KGWithCondRunner,
    LlamaConditionExtractor,
    LlamaEntityExtractor,
    OnlyLLMRunner,
    VectorRAGRunner,
)
from experiments.retrievers import (
    CachedUMLSMatcher,
    KGNoConditionsRetriever,
    KGWithConditionsRetriever,
    VectorRAGRetriever,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Trace log helper
# ──────────────────────────────────────────────────────────────────────

class TraceLog:
    def __init__(self, path: str):
        self.path = path
        self._fh: Optional[TextIO] = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh:
            self._fh.close()
            self._fh = None

    def write(self, text: str = ""):
        if self._fh:
            self._fh.write(text + "\n")
            self._fh.flush()

    def header(self, title: str, char: str = "="):
        self.write(char * 78)
        self.write(title)
        self.write(char * 78)

    def section(self, title: str):
        self.write("")
        self.write(f"── {title} " + "─" * max(2, 70 - len(title)))

    def kv(self, key: str, value):
        self.write(f"{key}: {value}")

    def block(self, label: str, body: str):
        self.write(f"[{label}]")
        for line in (body or "").splitlines() or [""]:
            self.write(f"  {line}")


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Full RAG experiment over a clinical QA dataset "
            "(default: MediQ); compares Only-LLM, Vector-RAG, "
            "KG-no-cond, and KG-with-cond using Llama-3.1-8B."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default="mediq",
                   help="Dataset name (currently supported: mediq)")
    p.add_argument("--dataset-path", default=None,
                   help="Path to the dataset jsonl (default: ExperimentConfig.dataset_path)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit number of samples (default: full dataset)")
    p.add_argument("--start-index", type=int, default=0,
                   help="Start sample index for resuming or sharding")
    p.add_argument("--variants", default="all",
                   help='Comma-separated variants or "all" '
                        f"(choices: {','.join(ALL_VARIANTS)})")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: $RAG_EXP_OUTPUT_DIR or ./output/full_rag_experiment)")
    p.add_argument("--llm-model", default=None,
                   help="HuggingFace model name "
                        "(default: $LLAMA_MODEL or meta-llama/Llama-3.1-8B-Instruct)")
    p.add_argument("--llm-dtype", default=None, choices=["bfloat16", "float16", "float32"])
    p.add_argument("--llm-attn-impl", default=None,
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--llm-batch-size", type=int, default=None,
                   help="Batched generation size for entity extraction")
    p.add_argument("--vector-top-k", type=int, default=None,
                   help="Top-k recommendations for Variant 2")
    p.add_argument("--no-deterministic", action="store_true",
                   help="Disable greedy decoding (use sampling)")
    p.add_argument("--no-trace", action="store_true",
                   help="Disable per-sample trace log")
    p.add_argument("--log-level", default="INFO")
    return p


def _resolve_variants(spec: str) -> tuple[str, ...]:
    if spec.strip().lower() == "all":
        return tuple(ALL_VARIANTS)
    chosen = [v.strip() for v in spec.split(",") if v.strip()]
    unknown = [v for v in chosen if v not in ALL_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown variant(s): {unknown}. Available: {ALL_VARIANTS}"
        )
    return tuple(chosen)


def _cfg_from_args(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if args.dataset_path:
        cfg.dataset_path = args.dataset_path
    cfg.dataset_name = args.dataset
    cfg.max_samples = args.max_samples
    cfg.start_index = args.start_index
    cfg.variants = _resolve_variants(args.variants)
    if args.output_dir:
        cfg.output_dir = args.output_dir
        cfg.cache_dir = os.path.join(args.output_dir, "cache")
    if args.llm_model:
        cfg.llm_model_name = args.llm_model
    if args.llm_dtype:
        cfg.llm_dtype = args.llm_dtype
    if args.llm_attn_impl:
        cfg.llm_attn_impl = args.llm_attn_impl
    if args.llm_batch_size is not None:
        cfg.llm_batch_size = args.llm_batch_size
    if args.vector_top_k is not None:
        cfg.vector_top_k = args.vector_top_k
    if args.no_deterministic:
        cfg.deterministic = False
    if args.no_trace:
        cfg.save_per_sample_trace = False
    cfg.log_level = args.log_level

    # cfg.__post_init__ has already run; redo dir creation if overridden.
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)
    return cfg


# ──────────────────────────────────────────────────────────────────────
# Setup: build LLM + retrievers + runners
# ──────────────────────────────────────────────────────────────────────

def _build_runners(cfg: ExperimentConfig) -> tuple[dict, dict]:
    """Returns (runners_by_name, shared_handles).

    shared_handles holds objects with flushable caches so the driver can persist
    them on shutdown.
    """
    runners: dict = {}
    handles: dict = {}

    # All four variants share one Llama backend.
    logger.info("Loading Llama-3.1 model…")
    llm = LocalLLM(
        model_name=cfg.llm_model_name,
        dtype=cfg.llm_dtype,
        attn_impl=cfg.llm_attn_impl,
        device=cfg.llm_device,
        hf_token=cfg.llm_hf_token,
        deterministic=cfg.deterministic,
    )
    handles["llm"] = llm

    if "only_llm" in cfg.variants:
        runners["only_llm"] = OnlyLLMRunner(llm, cfg)

    if "vector_rag" in cfg.variants:
        logger.info("Preparing Vector-RAG retriever (this loads/computes embeddings)…")
        vr = VectorRAGRetriever(cfg)
        handles["vector_retriever"] = vr
        runners["vector_rag"] = VectorRAGRunner(llm, cfg, vr)

    needs_entity = "kg_no_cond" in cfg.variants or "kg_with_cond" in cfg.variants
    if needs_entity:
        entity_extractor = LlamaEntityExtractor(llm, cfg)
        umls = CachedUMLSMatcher(cfg)
        handles["umls"] = umls

    needs_stage2 = "kg_no_cond" in cfg.variants
    needs_stage3 = "kg_with_cond" in cfg.variants

    # The Stage-2 and Stage-3 stores are independent JSON files; load only what's needed.
    if needs_stage2:
        logger.info("Loading Stage-2 KG (no conditions)…")
        kg_no_cond = KGNoConditionsRetriever(cfg)
        runners["kg_no_cond"] = KGNoCondRunner(
            llm, cfg, entity_extractor, umls, kg_no_cond,
        )

    if needs_stage3:
        logger.info("Loading Stage-3 KG (with conditions)…")
        kg_with_cond = KGWithConditionsRetriever(cfg)
        cond_extractor = LlamaConditionExtractor(llm, cfg)
        runners["kg_with_cond"] = KGWithCondRunner(
            llm, cfg, entity_extractor, cond_extractor, umls, kg_with_cond,
        )

    return runners, handles


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def _trace_sample(trace: TraceLog, sample: Sample, results: dict):
    trace.header(f"SAMPLE id={sample.sample_id}", char="=")
    trace.kv("question", sample.question)
    trace.block("context", "\n".join(sample.context) if isinstance(sample.context, list) else str(sample.context))
    trace.block("options", "\n".join(f"{k}. {v}" for k, v in sample.options.items()))
    trace.kv("gold_answer", sample.gold_answer)
    trace.kv("gold_answer_idx", sample.gold_answer_idx)
    for vname, r in results.items():
        trace.section(f"variant: {vname}")
        trace.kv("predicted", r.get("predicted", ""))
        trace.kv("correct", r.get("correct", False))
        trace.kv("elapsed_seconds", r.get("elapsed_seconds", 0))
        trace.kv("retrieval_strategy", r.get("retrieval_strategy", ""))
        trace.kv("retrieval_n_items", r.get("retrieval_n_items", 0))
        trace.kv(
            "tokens",
            f"retrieval={r.get('retrieval_tokens', 0)}  "
            f"answer_input={r.get('answer_input_tokens', 0)}  "
            f"answer_output={r.get('answer_output_tokens', 0)}  "
            f"answer_total={r.get('answer_total_tokens', 0)}",
        )
        if r.get("retrieval_text"):
            trace.block("retrieval_text", r["retrieval_text"])
        trace.block("llm_raw_response", r.get("raw_response", ""))


def _save_results(cfg: ExperimentConfig, dataset_name: str, results: list[dict],
                  per_variant_time: dict, total_elapsed: float):
    summary = _summarize(cfg, dataset_name, results, per_variant_time, total_elapsed)
    with open(cfg.results_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": summary, "results": results},
                  f, ensure_ascii=False, indent=2, default=str)
    with open(cfg.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def _summarize(cfg: ExperimentConfig, dataset_name: str, results: list[dict],
               per_variant_time: dict, total_elapsed: float) -> dict:
    per_variant = {}
    for v in cfg.variants:
        total = 0
        correct = 0
        elapsed_sum = 0.0
        errors = 0
        retr_tok_sum = 0
        in_tok_sum = 0
        out_tok_sum = 0
        for r in results:
            vr = (r.get("variants") or {}).get(v)
            if not vr:
                continue
            total += 1
            if vr.get("error"):
                errors += 1
                continue
            if vr.get("correct"):
                correct += 1
            elapsed_sum += vr.get("elapsed_seconds", 0.0)
            retr_tok_sum += vr.get("retrieval_tokens", 0)
            in_tok_sum += vr.get("answer_input_tokens", 0)
            out_tok_sum += vr.get("answer_output_tokens", 0)
        scored = max(total - errors, 1)
        per_variant[v] = {
            "samples_attempted": total,
            "errors": errors,
            "correct": correct,
            "accuracy": (correct / scored) if scored else 0.0,
            "total_elapsed_seconds": round(elapsed_sum, 3),
            "mean_elapsed_seconds": round(elapsed_sum / total, 4) if total else 0.0,
            "wall_clock_seconds": round(per_variant_time.get(v, 0.0), 3),
            # ── Token accounting (sum + mean across successful samples) ──
            "total_retrieval_tokens": retr_tok_sum,
            "total_answer_input_tokens": in_tok_sum,
            "total_answer_output_tokens": out_tok_sum,
            "total_answer_tokens": in_tok_sum + out_tok_sum,
            "mean_retrieval_tokens": round(retr_tok_sum / scored, 1),
            "mean_answer_input_tokens": round(in_tok_sum / scored, 1),
            "mean_answer_output_tokens": round(out_tok_sum / scored, 1),
            "mean_answer_tokens": round((in_tok_sum + out_tok_sum) / scored, 1),
        }

    return {
        "dataset": dataset_name,
        "n_samples": len(results),
        "variants": list(cfg.variants),
        "per_variant": per_variant,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "config": cfg.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }


def _flush_handles(handles: dict):
    vr = handles.get("vector_retriever")
    if vr is not None:
        try:
            vr.flush_cache()
        except Exception as e:
            logger.warning(f"Vector retriever flush failed: {e}")
    umls = handles.get("umls")
    if umls is not None:
        try:
            umls.flush()
        except Exception as e:
            logger.warning(f"UMLS cache flush failed: {e}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    setup_logging(args.log_level)
    cfg = _cfg_from_args(args)

    logger.info("=" * 72)
    logger.info("Full RAG Experiment")
    logger.info(f"  Dataset:   {cfg.dataset_name} ({cfg.dataset_path})")
    logger.info(f"  Variants:  {cfg.variants}")
    logger.info(f"  Model:     {cfg.llm_model_name}")
    logger.info(f"  Output:    {cfg.output_dir}")
    logger.info("=" * 72)

    dataset = load_dataset(
        cfg.dataset_name,
        path=cfg.dataset_path,
        max_samples=cfg.max_samples,
        start_index=cfg.start_index,
    )

    runners, handles = _build_runners(cfg)
    if not runners:
        logger.error("No runners were built; check --variants.")
        return 2

    per_variant_time: dict[str, float] = {v: 0.0 for v in cfg.variants}
    results: list[dict] = []

    trace_cm = TraceLog(cfg.trace_log_path) if cfg.save_per_sample_trace else _NullTrace()
    overall_start = time.perf_counter()
    try:
        with trace_cm as trace:
            trace.header("Full RAG Experiment trace", char="#")
            trace.kv("timestamp", datetime.now().isoformat())
            trace.kv("dataset", f"{cfg.dataset_name} ({cfg.dataset_path})")
            trace.kv("model", cfg.llm_model_name)
            trace.kv("variants", ", ".join(cfg.variants))
            trace.kv("n_samples", len(dataset))
            trace.write("")

            for i, sample in enumerate(dataset, 1):
                sample_record = {
                    "sample_id": sample.sample_id,
                    "question": sample.question,
                    "context": sample.context,
                    "options": sample.options,
                    "gold_answer": sample.gold_answer,
                    "gold_answer_idx": sample.gold_answer_idx,
                    "variants": {},
                }
                for vname in cfg.variants:
                    runner = runners.get(vname)
                    if runner is None:
                        continue
                    try:
                        r0 = time.perf_counter()
                        r = runner.run(sample)
                        per_variant_time[vname] += time.perf_counter() - r0
                    except Exception as e:
                        logger.exception(
                            f"Runner {vname!r} failed on sample {sample.sample_id}: {e}"
                        )
                        r = {
                            "predicted": "",
                            "correct": False,
                            "elapsed_seconds": 0.0,
                            "raw_response": "",
                            "retrieval_strategy": "error",
                            "retrieval_n_items": 0,
                            "retrieval_text": "",
                            "retrieval_debug": {},
                            "error": repr(e),
                        }
                    sample_record["variants"][vname] = r

                results.append(sample_record)
                if cfg.save_per_sample_trace:
                    _trace_sample(trace, sample, sample_record["variants"])

                if i % 10 == 0 or i == len(dataset):
                    flags = " | ".join(
                        f"{v}={_short_flag(sample_record['variants'].get(v, {}))}"
                        for v in cfg.variants
                    )
                    logger.info(f"[{i}/{len(dataset)}] sample={sample.sample_id}  {flags}")

                if cfg.flush_every and i % cfg.flush_every == 0:
                    total_elapsed = time.perf_counter() - overall_start
                    _save_results(cfg, cfg.dataset_name, results, per_variant_time, total_elapsed)
                    _flush_handles(handles)
                    logger.info(f"  ↳ incremental save at i={i}")
    finally:
        _flush_handles(handles)

    total_elapsed = time.perf_counter() - overall_start
    _save_results(cfg, cfg.dataset_name, results, per_variant_time, total_elapsed)

    # ── Console summary ──
    print()
    print("=" * 72)
    print(f"Total wall-clock: {total_elapsed:.1f}s   "
          f"Samples processed: {len(results)}")
    for v in cfg.variants:
        bucket = []
        for r in results:
            vr = (r.get("variants") or {}).get(v)
            if vr and not vr.get("error"):
                bucket.append(vr)
        if not bucket:
            continue
        n = len(bucket)
        correct = sum(1 for r in bucket if r.get("correct"))
        elapsed = sum(r.get("elapsed_seconds", 0) for r in bucket)
        mean_in_tok = sum(r.get("answer_input_tokens", 0) for r in bucket) / n
        mean_out_tok = sum(r.get("answer_output_tokens", 0) for r in bucket) / n
        print(
            f"  {v:<14}  acc={correct}/{n}={correct/n*100:.1f}%   "
            f"mean_e2e={elapsed/n:.3f}s   "
            f"mean_in_tok={mean_in_tok:.0f}   mean_out_tok={mean_out_tok:.0f}"
        )
    print("=" * 72)
    print(f"Results saved: {cfg.results_path}")
    print(f"Summary saved: {cfg.summary_path}")
    if cfg.save_per_sample_trace:
        print(f"Trace log:     {cfg.trace_log_path}")
    return 0


def _short_flag(variant_result: dict) -> str:
    if variant_result.get("error"):
        return "ERR"
    return "OK " if variant_result.get("correct") else "MIS"


class _NullTrace:
    """No-op stand-in when trace logging is disabled.

    Implements the same surface as TraceLog so the driver can call
    trace.header / trace.kv / etc. unconditionally without hasattr checks.
    """
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def write(self, *a, **kw): pass
    def header(self, *a, **kw): pass
    def section(self, *a, **kw): pass
    def kv(self, *a, **kw): pass
    def block(self, *a, **kw): pass


if __name__ == "__main__":
    sys.exit(main())
