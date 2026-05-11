"""Experiment configuration — values can be overridden via env vars or CLI."""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import config as project_config


ALL_VARIANTS = ("only_llm", "vector_rag", "kg_no_cond", "kg_with_cond")


@dataclass
class ExperimentConfig:
    # ── Dataset ──
    dataset_name: str = "mediq"
    dataset_path: str = os.path.join("./MediQ", "all_craft_md.jsonl")
    max_samples: Optional[int] = None       # None = full dataset
    start_index: int = 0                    # for resuming/sharding

    # ── Variants to run ──
    variants: tuple = ALL_VARIANTS

    # ── Local LLM (Llama-3.1-8B-Instruct) ──
    llm_model_name: str = os.environ.get(
        "LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
    )
    llm_dtype: str = "bfloat16"             # bfloat16 | float16
    llm_attn_impl: str = "sdpa"             # sdpa | flash_attention_2 | eager
    llm_device: str = "auto"                # "auto" | "cuda:0" | "cpu"
    llm_max_new_tokens_extraction: int = 1024
    llm_max_new_tokens_answer: int = 256
    llm_batch_size: int = 4                 # for batched generation
    llm_hf_token: Optional[str] = os.environ.get("HF_TOKEN", None)

    # ── OpenAI embeddings (Variant 2) ──
    openai_api_key: Optional[str] = os.environ.get(
        "OPENAI_API_KEY",
        getattr(project_config, "OPENAI_API_KEY", None),
    )
    embedding_model: str = "text-embedding-3-large"
    embedding_batch_size: int = 128
    vector_top_k: int = 5

    # ── UMLS REST API (Variant 3, 4 — entity matching) ──
    umls_api_key: Optional[str] = os.environ.get(
        "UMLS_API_KEY",
        getattr(project_config, "UMLS_API_KEY", None),
    )
    umls_max_workers: int = 4

    # ── KG sources ──
    stage0_recs_path: str = os.path.join(
        project_config.OUTPUT_DIR, project_config.OUTPUT_RECOMMENDATIONS_FILE
    )
    stage2_triples_path: str = os.path.join(
        project_config.OUTPUT_DIR, project_config.OUTPUT_TRIPLES_FILE
    )
    stage3_triples_path: str = os.path.join(
        project_config.OUTPUT_DIR, project_config.OUTPUT_AUGMENTED_TRIPLES_FILE
    )
    semantic_groups_file: str = project_config.SEMANTIC_GROUPS_FILE

    # ── KG retrieval limits ──
    kg_no_cond_one_hop_limit: int = 100
    kg_no_cond_paths_limit: int = 100
    kg_no_cond_paths_max_hops: int = 2
    kg_with_cond_31_limit: int = 50
    kg_with_cond_32_limit: int = 100
    kg_with_cond_33_limit: int = 5
    cond_keyword_min_len: int = 4
    cond_keyword_max: int = 30

    # ── Output ──
    output_dir: str = os.environ.get(
        "RAG_EXP_OUTPUT_DIR",
        os.path.join(project_config.OUTPUT_DIR, "full_rag_experiment"),
    )
    cache_dir: str = field(default="")      # set in __post_init__
    results_filename: str = "results.json"
    summary_filename: str = "summary.json"
    trace_log_filename: str = "trace.log"

    # ── Misc ──
    log_level: str = "INFO"
    deterministic: bool = True
    save_per_sample_trace: bool = True
    flush_every: int = 50                   # incremental save cadence

    def __post_init__(self):
        if not self.cache_dir:
            self.cache_dir = os.path.join(self.output_dir, "cache")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    # ── Convenience paths ──
    @property
    def results_path(self) -> str:
        return os.path.join(self.output_dir, self.results_filename)

    @property
    def summary_path(self) -> str:
        return os.path.join(self.output_dir, self.summary_filename)

    @property
    def trace_log_path(self) -> str:
        return os.path.join(self.output_dir, self.trace_log_filename)

    @property
    def embedding_cache_path(self) -> str:
        return os.path.join(self.cache_dir, "stage0_embeddings.npz")

    @property
    def query_embedding_cache_path(self) -> str:
        return os.path.join(self.cache_dir, "query_embeddings.npz")

    @property
    def umls_match_cache_path(self) -> str:
        return os.path.join(self.cache_dir, "umls_match_cache.json")

    def to_dict(self) -> dict:
        d = asdict(self)
        # Don't leak secrets into the saved summary.
        for k in ("openai_api_key", "umls_api_key", "llm_hf_token"):
            if d.get(k):
                d[k] = "***"
        return d
