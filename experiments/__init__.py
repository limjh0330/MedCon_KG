"""Full-dataset RAG experiment package.

Compares 4 RAG variants over a clinical QA dataset (default: MediQ):
  1. only_llm        — model answers from parametric knowledge only
  2. vector_rag      — OpenAI text-embedding-3-large cosine over Stage-0 recs
  3. kg_no_cond      — Llama entity extraction → UMLS CUI → Stage-2 KG
                       1-hop subgraph + 2-hop paths between matched CUIs
  4. kg_with_cond    — Llama entity + condition extraction → Stage-3 KG
                       3-1 → 3-2 → 3-3 cascade (mediq_graphrag_test.py logic)

All four variants share a single locally-loaded Llama-3.1-8B-Instruct model.
"""

from experiments.config import ExperimentConfig  # noqa: F401
