"""Build a persistent FAISS vector DB from Stage-0 raw texts.

Reads `stage0_documents.jsonl`, expands each document's `raw_texts` into
sentence-level records, embeds them with the same OpenAI backend used by
`VectorRAGRetriever`, and saves:

- `index.faiss`: normalized inner-product index (cosine similarity)
- `metadata.jsonl`: one JSON record per vector, same order as the FAISS rows
- `manifest.json`: build metadata and source paths

Example:
  python -m experiments.get_embed_db \
      --stage0-documents ./output/stage0_documents.jsonl \
      --output-dir ./output/vector_faiss_db \
      --db-prefixes CREST \
      --max-records 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli_utils import setup_logging
from experiments.config import ExperimentConfig
from experiments.llm_backend import OpenAIEmbedder

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a FAISS vector DB from Stage-0 raw_texts in "
            "stage0_documents.jsonl."
        )
    )
    parser.add_argument(
        "--stage0-documents",
        default=os.path.join(".", "output", "stage0_documents.jsonl"),
        help="Path to the Stage-0 documents JSONL.",
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
    parser.add_argument("--log-level", default="INFO")
    return parser


def _parse_prefixes(value: str) -> Optional[set[str]]:
    prefixes = {part.strip().upper() for part in (value or "").split(",") if part.strip()}
    return prefixes or None


def _infer_source_db(guideline_id: str) -> str:
    prefix, _, _ = str(guideline_id or "").partition("_")
    return prefix or ""


def _split_strength_prefixed_text(raw_text: str) -> tuple[str, str]:
    text = str(raw_text or "").strip()
    if text.startswith(":"):
        text = text[1:].strip()
    prefix, separator, remainder = text.partition(" : ")
    if (
        separator
        and prefix.strip()
        and remainder.strip()
        and len(prefix.strip()) <= 32
        and "\n" not in prefix
    ):
        return prefix.strip(), remainder.strip()
    return "", text


def _flush_batch(
    batch_rows: list[dict],
    *,
    embedder: OpenAIEmbedder,
    faiss_module,
    index,
    dimension: Optional[int],
    total_vectors: int,
    metadata_file,
) -> tuple[object, int, int]:
    vectors = embedder.embed_texts(
        [row["text"] for row in batch_rows]
    ).astype(np.float32, copy=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(batch_rows):
        raise ValueError(
            f"Embedding shape mismatch: got {vectors.shape}, expected ({len(batch_rows)}, D)"
        )

    if dimension is None:
        dimension = int(vectors.shape[1])
        index = faiss_module.IndexFlatIP(dimension)
        logger.info("Initialized FAISS IndexFlatIP with dimension=%s", dimension)

    faiss_module.normalize_L2(vectors)
    index.add(vectors)

    for offset, row in enumerate(batch_rows):
        row["vector_id"] = total_vectors + offset
        metadata_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_vectors += len(batch_rows)
    batch_rows.clear()
    return index, dimension, total_vectors


def _iter_filtered_records(
    stage0_documents_path: str,
    allowed_prefixes: Optional[set[str]],
    max_records: Optional[int],
    *,
    batch_size: int,
    embedder: OpenAIEmbedder,
    faiss_module,
    metadata_file,
    progress,
) -> tuple[object, Optional[int], int]:
    index = None
    dimension = None
    total_vectors = 0
    emitted = 0
    batch_rows: list[dict] = []

    with open(stage0_documents_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {stage0_documents_path}: {e}"
                ) from e

            guideline_id = str(
                doc.get("db_guideline_id")
                or doc.get("guideline_id")
                or ""
            ).strip()
            source_db = _infer_source_db(guideline_id)
            if allowed_prefixes and source_db.upper() not in allowed_prefixes:
                continue

            guideline_context = str(doc.get("guideline_context") or "")
            raw_texts = doc.get("raw_texts") or []
            if isinstance(raw_texts, str):
                raw_texts = [raw_texts]

            for sentence_index, raw_text in enumerate(raw_texts, start=1):
                raw_text = str(raw_text or "").strip()
                if not raw_text:
                    continue
                strength, text = _split_strength_prefixed_text(raw_text)
                if not text:
                    continue

                emitted += 1
                batch_rows.append({
                    "guideline_id": guideline_id,
                    "source_db": source_db,
                    "sentence_index": sentence_index,
                    "strength": strength,
                    "guideline_context": guideline_context,
                    "raw_text": raw_text,
                    "text": text,
                })

                if len(batch_rows) >= batch_size:
                    index, dimension, total_vectors = _flush_batch(
                        batch_rows,
                        embedder=embedder,
                        faiss_module=faiss_module,
                        index=index,
                        dimension=dimension,
                        total_vectors=total_vectors,
                        metadata_file=metadata_file,
                    )
                    progress.update(batch_size)

                if max_records is not None and emitted >= max_records:
                    break

            if max_records is not None and emitted >= max_records:
                break

    if batch_rows:
        final_batch_size = len(batch_rows)
        index, dimension, total_vectors = _flush_batch(
            batch_rows,
            embedder=embedder,
            faiss_module=faiss_module,
            index=index,
            dimension=dimension,
            total_vectors=total_vectors,
            metadata_file=metadata_file,
        )
        progress.update(final_batch_size)

    return index, dimension, total_vectors


def build_faiss_db(args: argparse.Namespace) -> dict:
    import faiss

    cfg = ExperimentConfig()
    if args.embedding_model:
        cfg.embedding_model = args.embedding_model
    if args.batch_size is not None:
        cfg.embedding_batch_size = args.batch_size

    if not os.path.isfile(args.stage0_documents):
        raise FileNotFoundError(
            f"Stage-0 documents JSONL not found: {args.stage0_documents}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    metadata_jsonl_path = os.path.join(args.output_dir, "metadata.jsonl")
    faiss_index_path = os.path.join(args.output_dir, "index.faiss")
    manifest_path = os.path.join(args.output_dir, "manifest.json")

    documents_jsonl_path = os.path.abspath(args.stage0_documents)
    allowed_prefixes = _parse_prefixes(args.db_prefixes)
    logger.info(
        "Streaming Stage-0 records with prefixes: %s%s",
        allowed_prefixes or "ALL",
        f", max_records={args.max_records}" if args.max_records is not None else "",
    )

    embedder = OpenAIEmbedder(
        api_key=cfg.openai_api_key,
        model=cfg.embedding_model,
        batch_size=cfg.embedding_batch_size,
    )

    with open(metadata_jsonl_path, "w", encoding="utf-8") as meta_f:
        progress = tqdm(desc="Indexing records", unit="record", dynamic_ncols=True)
        try:
            index, dimension, total_vectors = _iter_filtered_records(
                stage0_documents_path=args.stage0_documents,
                allowed_prefixes=allowed_prefixes,
                max_records=args.max_records,
                batch_size=cfg.embedding_batch_size,
                embedder=embedder,
                faiss_module=faiss,
                metadata_file=meta_f,
                progress=progress,
            )
        finally:
            progress.close()

    if index is None or dimension is None:
        raise ValueError("No records were indexed. Check the Stage-0 input or filters.")

    faiss.write_index(index, faiss_index_path)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "stage0_documents_path": documents_jsonl_path,
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
