"""Build a persistent FAISS vector DB from Stage-0 raw texts.

Reads the Stage-0 metadata JSON (for example `output/stage0_recommendations.json`),
resolves the underlying `documents_jsonl`, expands each document's `raw_texts`
into sentence-level records, embeds them with the same OpenAI backend used by
`VectorRAGRetriever`, and saves:

- `index.faiss`: normalized inner-product index (cosine similarity)
- `metadata.jsonl`: one JSON record per vector, same order as the FAISS rows
- `manifest.json`: build metadata and source paths

Example:
  python -m experiments.get_embed_db \
      --stage0-meta ./output/stage0_recommendations.json \
      --output-dir ./output/vector_faiss_db \
      --db-prefixes CREST \
      --max-records 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Iterable, Iterator, Optional

import numpy as np
from tqdm import tqdm

from cli_utils import setup_logging
from experiments.config import ExperimentConfig
from experiments.llm_backend import OpenAIEmbedder
from experiments.retrievers import iter_stage0_recommendation_records

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a FAISS vector DB from Stage-0 raw_texts referenced by "
            "stage0_recommendations.json."
        )
    )
    parser.add_argument(
        "--stage0-meta",
        default=os.path.join(".", "output", "stage0_recommendations.json"),
        help="Path to the Stage-0 metadata JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(".", "output", "vector_faiss_db"),
        help="Directory to store the FAISS index and metadata.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="OpenAI embedding model override (default: ExperimentConfig.embedding_model).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding batch size override (default: ExperimentConfig.embedding_batch_size).",
    )
    parser.add_argument(
        "--db-prefixes",
        default="",
        help=(
            "Comma-separated guideline ID prefixes to include, for example "
            "`CREST` or `PUBMED`. Empty means all records."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional hard limit on sentence-level records to index.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse records and report counts without calling the embedding API.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _parse_prefixes(value: str) -> Optional[set[str]]:
    prefixes = {part.strip().upper() for part in (value or "").split(",") if part.strip()}
    return prefixes or None


def _infer_source_db(guideline_id: str) -> str:
    prefix, _, _ = str(guideline_id or "").partition("_")
    return prefix or ""


def _iter_filtered_records(
    stage0_meta_path: str,
    allowed_prefixes: Optional[set[str]],
    max_records: Optional[int],
) -> Iterator[dict]:
    emitted = 0
    for rec in iter_stage0_recommendation_records(stage0_meta_path):
        guideline_id = str(
            rec.get("guideline_id")
            or rec.get("db_guideline_id")
            or ""
        ).strip()
        source_db = _infer_source_db(guideline_id)
        if allowed_prefixes and source_db.upper() not in allowed_prefixes:
            continue

        text = str(
            rec.get("text")
            or rec.get("raw_text")
            or rec.get("abstract")
            or rec.get("guideline_context")
            or ""
        ).strip()
        if not text:
            continue

        emitted += 1
        yield {
            "guideline_id": guideline_id,
            "source_db": source_db,
            "sentence_index": rec.get("sentence_index"),
            "strength": rec.get("strength", ""),
            "raw_text": str(rec.get("raw_text") or text),
            "text": text,
        }
        if max_records is not None and emitted >= max_records:
            break


def _batched(iterable: Iterable[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_stage0_meta(stage0_meta_path: str) -> dict:
    with open(stage0_meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_documents_jsonl(stage0_meta_path: str, stage0_meta: dict) -> Optional[str]:
    rel = (
        stage0_meta.get("documents_jsonl")
        or stage0_meta.get("document_jsonl")
        or stage0_meta.get("stage0_documents_jsonl")
    )
    if not rel:
        return None
    if os.path.isabs(rel):
        return rel
    return os.path.join(os.path.dirname(stage0_meta_path), rel)


def build_faiss_db(args: argparse.Namespace) -> dict:
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "FAISS is required. Install `faiss-cpu` or another compatible FAISS build."
        ) from e

    cfg = ExperimentConfig()
    if args.embedding_model:
        cfg.embedding_model = args.embedding_model
    if args.batch_size is not None:
        cfg.embedding_batch_size = args.batch_size

    if not os.path.isfile(args.stage0_meta):
        raise FileNotFoundError(f"Stage-0 metadata JSON not found: {args.stage0_meta}")

    os.makedirs(args.output_dir, exist_ok=True)
    metadata_jsonl_path = os.path.join(args.output_dir, "metadata.jsonl")
    faiss_index_path = os.path.join(args.output_dir, "index.faiss")
    manifest_path = os.path.join(args.output_dir, "manifest.json")

    stage0_meta = _load_stage0_meta(args.stage0_meta)
    documents_jsonl_path = _resolve_documents_jsonl(args.stage0_meta, stage0_meta)
    allowed_prefixes = _parse_prefixes(args.db_prefixes)

    record_iter = _iter_filtered_records(
        stage0_meta_path=args.stage0_meta,
        allowed_prefixes=allowed_prefixes,
        max_records=args.max_records,
    )

    if args.dry_run:
        sample_records: list[dict] = []
        total = 0
        for rec in record_iter:
            total += 1
            if len(sample_records) < 3:
                sample_records.append(rec)
        summary = {
            "mode": "dry-run",
            "record_count": total,
            "sample_records": sample_records,
            "stage0_meta_path": os.path.abspath(args.stage0_meta),
            "documents_jsonl_path": (
                os.path.abspath(documents_jsonl_path) if documents_jsonl_path else None
            ),
        }
        logger.info("Dry-run parsed %s sentence-level records", total)
        return summary

    embedder = OpenAIEmbedder(
        api_key=cfg.openai_api_key,
        model=cfg.embedding_model,
        batch_size=cfg.embedding_batch_size,
    )

    index = None
    total_vectors = 0
    dimension = None

    with open(metadata_jsonl_path, "w", encoding="utf-8") as meta_f:
        progress = tqdm(desc="Indexing records", unit="record", dynamic_ncols=True)
        for batch in _batched(record_iter, cfg.embedding_batch_size):
            texts = [item["text"] for item in batch]
            vectors = embedder.embed_texts(texts).astype(np.float32, copy=False)
            if vectors.ndim != 2 or vectors.shape[0] != len(batch):
                raise ValueError(
                    f"Embedding shape mismatch: got {vectors.shape}, expected ({len(batch)}, D)"
                )

            if dimension is None:
                dimension = int(vectors.shape[1])
                index = faiss.IndexFlatIP(dimension)
                logger.info("Initialized FAISS IndexFlatIP with dimension=%s", dimension)

            faiss.normalize_L2(vectors)
            index.add(vectors)

            for offset, item in enumerate(batch):
                row = {
                    "vector_id": total_vectors + offset,
                    "guideline_id": item["guideline_id"],
                    "source_db": item["source_db"],
                    "sentence_index": item["sentence_index"],
                    "strength": item["strength"],
                    "text": item["text"],
                    "raw_text": item["raw_text"],
                }
                meta_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            total_vectors += len(batch)
            progress.update(len(batch))

        progress.close()

    if index is None or dimension is None:
        raise ValueError("No records were indexed. Check the Stage-0 input or filters.")

    faiss.write_index(index, faiss_index_path)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "stage0_meta_path": os.path.abspath(args.stage0_meta),
        "documents_jsonl_path": (
            os.path.abspath(documents_jsonl_path) if documents_jsonl_path else None
        ),
        "output_dir": os.path.abspath(args.output_dir),
        "faiss_index_path": os.path.abspath(faiss_index_path),
        "metadata_jsonl_path": os.path.abspath(metadata_jsonl_path),
        "embedding_model": cfg.embedding_model,
        "embedding_batch_size": cfg.embedding_batch_size,
        "faiss_metric": "cosine_similarity_via_normalized_inner_product",
        "dimension": dimension,
        "record_count": total_vectors,
        "max_records": args.max_records,
        "db_prefixes": sorted(allowed_prefixes) if allowed_prefixes else [],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("Saved FAISS index to %s", faiss_index_path)
    logger.info("Saved metadata JSONL to %s", metadata_jsonl_path)
    logger.info("Saved manifest to %s", manifest_path)
    return manifest


def main():
    args = _build_parser().parse_args()
    setup_logging(args.log_level)
    result = build_faiss_db(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
