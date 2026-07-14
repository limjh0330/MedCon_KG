"""Build a sharded FAISS vector DB from Stage-0 raw texts.

Reads `stage0_documents.jsonl`, expands each document's `raw_texts` into
sentence-level records, embeds them with the OpenAI backend used by the
retriever, and saves:

- `index_1024d_shard_*.faiss`: normalized inner-product shards
- `metadata_1024d.jsonl`: one JSON record per vector, same order as FAISS rows
- `manifest_1024d.json`: build metadata and shard inventory

Resume is supported. Existing metadata is read first, previously indexed keys
are skipped, and the current shard is checkpointed every 10,000 new vectors.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from glob import glob
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cli_utils import setup_logging
from llm_backend import OpenAIEmbedder

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIMENSION = 1024
CHECKPOINT_INTERVAL = 10_000
SHARD_SIZE = 100_000
SHARD_RE = re.compile(r"^index_(\d+)d_shard_(\d{6})\.faiss$")


@dataclass
class ArtifactPaths:
    output_dir: str
    metadata_path: str
    manifest_path: str
    dimension: int

    def shard_path(self, shard_id: int) -> str:
        return os.path.join(
            self.output_dir,
            f"index_{self.dimension}d_shard_{shard_id:06d}.faiss",
        )


@dataclass
class ShardState:
    shard_id: int
    index: object | None
    vector_count: int
    last_checkpoint_count: int


def _load_env_file() -> None:
    """Load `.env` when python-dotenv is available, but do not require it."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sharded FAISS vector DB from Stage-0 raw_texts in "
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
        help="Directory to store the FAISS shards and metadata.",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model name.",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSION,
        help="Embedding dimension override for text-embedding-3 models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
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
    prefixes = {
        part.strip().upper() for part in (value or "").split(",") if part.strip()
    }
    return prefixes or None


def _artifact_paths(output_dir: str, embedding_dimension: int) -> ArtifactPaths:
    output_dir = os.path.abspath(output_dir)
    suffix = f"{embedding_dimension}d"
    return ArtifactPaths(
        output_dir=output_dir,
        metadata_path=os.path.join(output_dir, f"metadata_{suffix}.jsonl"),
        manifest_path=os.path.join(output_dir, f"manifest_{suffix}.json"),
        dimension=embedding_dimension,
    )


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


def _iter_sentence_rows(
    stage0_documents_path: str,
    allowed_prefixes: Optional[set[str]],
    max_records: Optional[int],
):
    emitted = 0

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
                doc.get("db_guideline_id") or doc.get("guideline_id") or ""
            ).strip()
            source_db = _infer_source_db(guideline_id)
            if allowed_prefixes and source_db.upper() not in allowed_prefixes:
                continue

            raw_texts = doc.get("raw_texts") or []
            if isinstance(raw_texts, str):
                raw_texts = [raw_texts]

            guideline_context = str(doc.get("guideline_context") or "")
            for sentence_index, raw_text in enumerate(raw_texts, start=1):
                raw_text = str(raw_text or "").strip()
                if not raw_text:
                    continue

                strength, text = _split_strength_prefixed_text(raw_text)
                if not text:
                    continue

                yield {
                    "guideline_id": guideline_id,
                    "source_db": source_db,
                    "sentence_index": sentence_index,
                    "strength": strength,
                    "guideline_context": guideline_context,
                    "raw_text": raw_text,
                    "text": text,
                }
                emitted += 1
                if max_records is not None and emitted >= max_records:
                    return


def _record_key_from_row(row: dict) -> tuple[str, str, int]:
    return (
        str(row.get("guideline_id") or "").strip(),
        str(row.get("source_db") or "").strip(),
        int(row.get("sentence_index") or 0),
    )


def _count_target_sentences(
    stage0_documents_path: str,
    allowed_prefixes: Optional[set[str]],
    max_records: Optional[int],
    existing_record_keys: set[tuple[str, str, int]],
) -> tuple[int, int]:
    total_sentences = 0
    already_indexed_sentences = 0

    for row in _iter_sentence_rows(
        stage0_documents_path=stage0_documents_path,
        allowed_prefixes=allowed_prefixes,
        max_records=max_records,
    ):
        total_sentences += 1
        if _record_key_from_row(row) in existing_record_keys:
            already_indexed_sentences += 1

    return total_sentences, already_indexed_sentences


def _cleanup_temp_files(output_path: str) -> None:
    for temp_path in glob(f"{output_path}.tmp.*"):
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _write_index_checkpoint(faiss_module, index, output_path: str) -> None:
    """Persist a shard checkpoint without keeping a parallel tmp shard file."""
    _cleanup_temp_files(output_path)
    faiss_module.write_index(index, output_path)
    temp_path = f"{output_path}.tmp.{os.getpid()}"
    if os.path.exists(temp_path):
        os.remove(temp_path)


def _load_existing_metadata_keys(
    metadata_path: str,
) -> tuple[set[tuple[str, str, int]], int]:
    if not os.path.isfile(metadata_path):
        return set(), 0

    keys: set[tuple[str, str, int]] = set()
    count = 0
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {metadata_path}: {e}"
                ) from e

            vector_id = row.get("vector_id")
            if vector_id != count:
                raise ValueError(
                    f"Non-contiguous vector_id on line {line_no} of "
                    f"{metadata_path}: got {vector_id}, expected {count}"
                )
            keys.add(_record_key_from_row(row))
            count += 1
    return keys, count


def _truncate_metadata_to_count(metadata_path: str, target_count: int) -> None:
    kept = 0
    with open(metadata_path, "r+b") as f:
        while kept < target_count:
            line = f.readline()
            if not line:
                raise ValueError(
                    f"Metadata ended at {kept} records; cannot truncate to {target_count}."
                )
            if line.strip():
                kept += 1
        f.truncate(f.tell())


def _parse_shard_path(path: str, embedding_dimension: int) -> int:
    match = SHARD_RE.match(os.path.basename(path))
    if not match:
        raise ValueError(f"Unexpected shard filename: {path}")

    file_dimension = int(match.group(1))
    if file_dimension != embedding_dimension:
        raise ValueError(
            f"Shard dimension mismatch in filename {path}: "
            f"got {file_dimension}, expected {embedding_dimension}"
        )
    return int(match.group(2))


def _list_shard_paths(output_dir: str, embedding_dimension: int) -> list[str]:
    paths = [
        str(path)
        for path in Path(output_dir).glob(f"index_{embedding_dimension}d_shard_*.faiss")
    ]
    return sorted(paths, key=lambda path: _parse_shard_path(path, embedding_dimension))


def _load_existing_shards(faiss_module, paths: ArtifactPaths) -> tuple[list[dict], ShardState]:
    shard_paths = _list_shard_paths(paths.output_dir, paths.dimension)
    shard_infos: list[dict] = []
    active_state = ShardState(shard_id=0, index=None, vector_count=0, last_checkpoint_count=0)

    for expected_id, shard_path in enumerate(shard_paths):
        shard_id = _parse_shard_path(shard_path, paths.dimension)
        if shard_id != expected_id:
            raise ValueError(
                f"Shard sequence gap detected: expected shard {expected_id:06d}, "
                f"found {shard_id:06d}."
            )

        index = faiss_module.read_index(shard_path)
        dimension = int(index.d)
        vector_count = int(index.ntotal)
        if dimension != paths.dimension:
            raise ValueError(
                f"Shard dimension mismatch in {shard_path}: "
                f"got {dimension}, expected {paths.dimension}"
            )
        if vector_count <= 0:
            raise ValueError(f"Shard {shard_path} is empty.")
        if expected_id < len(shard_paths) - 1 and vector_count != SHARD_SIZE:
            raise ValueError(
                f"Non-final shard {shard_path} has {vector_count} vectors; "
                f"expected exactly {SHARD_SIZE}."
            )

        shard_infos.append(
            {
                "shard_id": shard_id,
                "path": shard_path,
                "vector_count": vector_count,
                "dimension": dimension,
            }
        )

        if expected_id == len(shard_paths) - 1 and vector_count < SHARD_SIZE:
            active_state = ShardState(
                shard_id=shard_id,
                index=index,
                vector_count=vector_count,
                last_checkpoint_count=vector_count,
            )
        else:
            active_state = ShardState(
                shard_id=shard_id + 1,
                index=None,
                vector_count=0,
                last_checkpoint_count=0,
            )

    return shard_infos, active_state


def _persist_manifest(
    manifest_path: str,
    manifest: dict,
) -> None:
    temp_path = f"{manifest_path}.tmp.{os.getpid()}"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, manifest_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _update_shard_inventory(
    shard_infos: list[dict],
    paths: ArtifactPaths,
    shard_id: int,
    vector_count: int,
) -> None:
    shard_path = paths.shard_path(shard_id)
    entry = {
        "shard_id": shard_id,
        "path": shard_path,
        "vector_count": vector_count,
        "dimension": paths.dimension,
    }
    for idx, existing in enumerate(shard_infos):
        if existing["shard_id"] == shard_id:
            shard_infos[idx] = entry
            return
    shard_infos.append(entry)
    shard_infos.sort(key=lambda item: item["shard_id"])


def _checkpoint_current_shard(
    faiss_module,
    paths: ArtifactPaths,
    state: ShardState,
    shard_infos: list[dict],
) -> None:
    if state.index is None or state.vector_count == 0:
        return
    _write_index_checkpoint(faiss_module, state.index, paths.shard_path(state.shard_id))
    _update_shard_inventory(shard_infos, paths, state.shard_id, state.vector_count)
    state.last_checkpoint_count = state.vector_count


def _flush_batch(
    batch_rows: list[dict],
    *,
    embedder: OpenAIEmbedder,
    faiss_module,
    paths: ArtifactPaths,
    metadata_file,
    state: ShardState,
    shard_infos: list[dict],
    dimension: Optional[int],
    total_vectors: int,
) -> tuple[ShardState, int, int]:
    vectors = embedder.embed_texts([row["text"] for row in batch_rows]).astype(
        np.float32,
        copy=False,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(batch_rows):
        raise ValueError(
            f"Embedding shape mismatch: got {vectors.shape}, expected ({len(batch_rows)}, D)"
        )

    batch_dimension = int(vectors.shape[1])
    expected_dimension = dimension or paths.dimension
    if batch_dimension != expected_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: got {batch_dimension}, "
            f"expected {expected_dimension}"
        )

    faiss_module.normalize_L2(vectors)

    row_offset = 0
    while row_offset < len(batch_rows):
        if state.index is None:
            state = ShardState(
                shard_id=state.shard_id,
                index=faiss_module.IndexFlatIP(expected_dimension),
                vector_count=0,
                last_checkpoint_count=0,
            )
            logger.info(
                "Initialized shard %s with dimension=%s",
                f"{state.shard_id:06d}",
                expected_dimension,
            )

        shard_remaining = SHARD_SIZE - state.vector_count
        take = min(shard_remaining, len(batch_rows) - row_offset)
        next_offset = row_offset + take

        state.index.add(vectors[row_offset:next_offset])

        shard_base = state.vector_count
        for local_offset, row in enumerate(batch_rows[row_offset:next_offset]):
            row["vector_id"] = total_vectors
            row["shard_id"] = state.shard_id
            row["shard_vector_id"] = shard_base + local_offset
            metadata_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_vectors += 1

        state.vector_count += take
        row_offset = next_offset

        if (
            state.vector_count - state.last_checkpoint_count >= CHECKPOINT_INTERVAL
            or state.vector_count == SHARD_SIZE
        ):
            _checkpoint_current_shard(
                faiss_module=faiss_module,
                paths=paths,
                state=state,
                shard_infos=shard_infos,
            )
            logger.info(
                "Checkpointed shard %s at %s vectors",
                f"{state.shard_id:06d}",
                state.vector_count,
            )

        if state.vector_count == SHARD_SIZE:
            logger.info(
                "Completed shard %s with %s vectors",
                f"{state.shard_id:06d}",
                state.vector_count,
            )
            state = ShardState(
                shard_id=state.shard_id + 1,
                index=None,
                vector_count=0,
                last_checkpoint_count=0,
            )

    metadata_file.flush()
    os.fsync(metadata_file.fileno())
    return state, expected_dimension, total_vectors


def build_faiss_db(args: argparse.Namespace) -> dict:
    import faiss

    _load_env_file()

    stage0_documents_path = os.path.abspath(args.stage0_documents)
    if not os.path.isfile(stage0_documents_path):
        raise FileNotFoundError(
            f"Stage-0 documents JSONL not found: {stage0_documents_path}"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.embedding_dimension <= 0:
        raise ValueError("--embedding-dimension must be a positive integer.")

    allowed_prefixes = _parse_prefixes(args.db_prefixes)

    os.makedirs(args.output_dir, exist_ok=True)
    paths = _artifact_paths(args.output_dir, args.embedding_dimension)

    existing_record_keys: set[tuple[str, str, int]] = set()
    existing_record_count = 0
    if os.path.isfile(paths.metadata_path):
        existing_record_keys, existing_record_count = _load_existing_metadata_keys(
            paths.metadata_path
        )

    shard_infos, state = _load_existing_shards(faiss, paths)
    existing_vector_count = sum(info["vector_count"] for info in shard_infos)

    if existing_vector_count and not os.path.isfile(paths.metadata_path):
        raise FileNotFoundError(
            "Existing shard files were found but the matching metadata file is missing: "
            f"{paths.metadata_path}"
        )
    if existing_record_count > existing_vector_count:
        logger.warning(
            "Metadata is ahead of the last valid FAISS checkpoint "
            "(metadata=%s, shards=%s); rolling metadata back.",
            existing_record_count,
            existing_vector_count,
        )
        _truncate_metadata_to_count(paths.metadata_path, existing_vector_count)
        existing_record_keys, existing_record_count = _load_existing_metadata_keys(
            paths.metadata_path
        )
    elif existing_vector_count > existing_record_count:
        raise ValueError(
            "Existing shard/metadata count mismatch: "
            f"shards={existing_vector_count}, metadata={existing_record_count}. "
            "The shards are ahead of metadata and cannot be recovered automatically."
        )

    if os.path.isfile(paths.manifest_path):
        with open(paths.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_dimension = int(manifest.get("dimension") or args.embedding_dimension)
        if manifest_dimension != args.embedding_dimension:
            raise ValueError(
                f"Existing manifest dimension is {manifest_dimension}, "
                f"but requested dimension is {args.embedding_dimension}."
            )

    total_target_sentences, already_indexed_sentences = _count_target_sentences(
        stage0_documents_path=stage0_documents_path,
        allowed_prefixes=allowed_prefixes,
        max_records=args.max_records,
        existing_record_keys=existing_record_keys,
    )
    if total_target_sentences == 0:
        raise ValueError("No records were indexed. Check the Stage-0 input or filters.")

    logger.info(
        "Streaming Stage-0 records with prefixes: %s%s",
        allowed_prefixes or "ALL",
        f", max_records={args.max_records}" if args.max_records is not None else "",
    )
    logger.info(
        "Resume baseline: %s/%s eligible sentences already indexed",
        already_indexed_sentences,
        total_target_sentences,
    )

    embedder = OpenAIEmbedder(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=args.embedding_model,
        dimensions=args.embedding_dimension,
        batch_size=args.batch_size,
    )

    metadata_mode = "a" if existing_record_count else "w"
    batch_rows: list[dict] = []
    total_vectors = existing_record_count
    dimension = args.embedding_dimension if existing_record_count else None
    skipped_existing = 0

    with open(paths.metadata_path, metadata_mode, encoding="utf-8") as metadata_file:
        progress = tqdm(
            total=total_target_sentences,
            initial=already_indexed_sentences,
            desc="Embedding sentences",
            unit="sentence",
            dynamic_ncols=True,
        )
        try:
            for row in _iter_sentence_rows(
                stage0_documents_path=stage0_documents_path,
                allowed_prefixes=allowed_prefixes,
                max_records=args.max_records,
            ):
                key = _record_key_from_row(row)
                if key in existing_record_keys:
                    skipped_existing += 1
                    continue

                batch_rows.append(row)
                if len(batch_rows) < args.batch_size:
                    continue

                flushed_count = len(batch_rows)
                state, dimension, total_vectors = _flush_batch(
                    batch_rows,
                    embedder=embedder,
                    faiss_module=faiss,
                    paths=paths,
                    metadata_file=metadata_file,
                    state=state,
                    shard_infos=shard_infos,
                    dimension=dimension,
                    total_vectors=total_vectors,
                )
                progress.update(flushed_count)
                batch_rows.clear()

            if batch_rows:
                flushed_count = len(batch_rows)
                state, dimension, total_vectors = _flush_batch(
                    batch_rows,
                    embedder=embedder,
                    faiss_module=faiss,
                    paths=paths,
                    metadata_file=metadata_file,
                    state=state,
                    shard_infos=shard_infos,
                    dimension=dimension,
                    total_vectors=total_vectors,
                )
                progress.update(flushed_count)
                batch_rows.clear()
        finally:
            progress.close()

    if state.index is not None and state.vector_count > state.last_checkpoint_count:
        _checkpoint_current_shard(
            faiss_module=faiss,
            paths=paths,
            state=state,
            shard_infos=shard_infos,
        )
        logger.info(
            "Final checkpointed shard %s at %s vectors",
            f"{state.shard_id:06d}",
            state.vector_count,
        )

    if total_vectors == 0 or dimension is None:
        raise ValueError("No records were indexed. Check the Stage-0 input or filters.")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "stage0_documents_path": stage0_documents_path,
        "output_dir": paths.output_dir,
        "metadata_jsonl_path": paths.metadata_path,
        "manifest_path": paths.manifest_path,
        "embedding_model": args.embedding_model,
        "embedding_dimension": args.embedding_dimension,
        "embedding_batch_size": args.batch_size,
        "faiss_metric": "cosine_similarity_via_normalized_inner_product",
        "dimension": dimension,
        "record_count": total_vectors,
        "existing_record_count": existing_record_count,
        "new_record_count": total_vectors - existing_record_count,
        "skipped_existing_record_count": skipped_existing,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "shard_size": SHARD_SIZE,
        "max_records": args.max_records,
        "db_prefixes": sorted(allowed_prefixes) if allowed_prefixes else [],
        "shards": shard_infos,
    }
    _persist_manifest(paths.manifest_path, manifest)

    logger.info("Saved metadata JSONL to %s", paths.metadata_path)
    logger.info("Saved manifest to %s", paths.manifest_path)
    logger.info("Saved %s shard file(s)", len(shard_infos))
    return manifest


def main() -> None:
    args = _build_parser().parse_args()
    setup_logging(args.log_level)
    result = build_faiss_db(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
