"""Dataset abstractions.

Adding a new dataset later:
  1. Subclass BaseDataset and implement __iter__/__len__.
  2. Register it in DATASET_REGISTRY.

Every dataset emits Samples in the same shape so downstream runners are
dataset-agnostic. Datasets without options (open-ended QA) can leave the
options dict empty — runners handle the empty case.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    sample_id: str
    question: str
    context: list[str] = field(default_factory=list)
    options: dict[str, str] = field(default_factory=dict)
    gold_answer: str = ""
    gold_answer_idx: str = ""
    raw: dict = field(default_factory=dict)

    def joined_context(self) -> str:
        if isinstance(self.context, list):
            return " ".join(s.strip() for s in self.context if s and s.strip())
        return str(self.context).strip()

    def input_query(self) -> str:
        """The verbatim text fed to RAG retrievers (context above, question below)."""
        return f"[CONTEXT]\n{self.joined_context()}\n\n[QUESTION]\n{self.question.strip()}"


class BaseDataset(ABC):
    name: str = "base"

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]: ...

    @abstractmethod
    def __len__(self) -> int: ...


class MediQDataset(BaseDataset):
    """MediQ-style jsonl with fields: id, question, context[], options{}, answer, answer_idx."""

    name = "mediq"

    def __init__(
        self,
        path: str,
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.path = path
        self.max_samples = max_samples
        self.start_index = start_index
        self._rows: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._rows.append(json.loads(line))
        logger.info(f"Loaded {len(self._rows)} rows from {path}")

        end = (
            min(start_index + max_samples, len(self._rows))
            if max_samples is not None
            else len(self._rows)
        )
        self._slice = self._rows[start_index:end]
        logger.info(
            f"Effective sample range: [{start_index}, {end}) "
            f"→ {len(self._slice)} samples"
        )

    def __iter__(self) -> Iterator[Sample]:
        for r in self._slice:
            yield Sample(
                sample_id=str(r.get("id", "")),
                question=r.get("question", "") or "",
                context=r.get("context", []) or [],
                options=r.get("options", {}) or {},
                gold_answer=r.get("answer", "") or "",
                gold_answer_idx=r.get("answer_idx", "") or "",
                raw=r,
            )

    def __len__(self) -> int:
        return len(self._slice)


# ── Registry for runtime dataset selection ──
DATASET_REGISTRY = {
    "mediq": MediQDataset,
}


def load_dataset(name: str, **kwargs) -> BaseDataset:
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name!r}. Available: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name](**kwargs)
