"""Llama-based extractors + 4 variant runners.

All four runners share one LocalLLM instance. Each runner.run(sample) does the
variant-specific work end-to-end and returns a record dict.

Per-sample end-to-end timing is measured inside run(); it covers extraction,
retrieval, and answer generation. Setup costs (model load, embedding
precompute, KG load) are excluded.
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Optional

from experiments.config import ExperimentConfig
from experiments.datasets import Sample
from experiments.llm_backend import BaseLLM, parse_json_object
from experiments.retrievers import (
    CachedUMLSMatcher,
    KGNoConditionsRetriever,
    KGWithConditionsRetriever,
    RetrievalResult,
    VectorRAGRetriever,
)

logger = logging.getLogger(__name__)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\[(CONTEXT|QUESTION)\]\s*", "", text)
    parts = _SENTENCE_SPLIT_RE.split(cleaned.strip())
    return [p.strip() for p in parts if p and p.strip()]


# ──────────────────────────────────────────────────────────────────────
# Prompts — kept compact for an 8B model
# ──────────────────────────────────────────────────────────────────────

ENTITY_SYSTEM_PROMPT = """You are a medical Named Entity Recognition specialist. From a clinical sentence, extract every medical entity candidate that could be mapped to UMLS.

For each entity emit:
- surface_form: exact phrase from the input
- normalized_form: UMLS-style preferred English term (if same as surface_form, repeat it)
- semantic_group: one of ACTI, ANAT, CHEM, CONC, DEVI, DISO, GENE, GEOG, LIVB, OBJC, OCCU, ORGA, PHEN, PHYS, PROC

Rules:
- Prioritize recall. When in doubt, include the entity.
- Extract multi-word terms as full phrases.
- Also extract clinically meaningful sub-components (e.g. "platinum-based chemotherapy" → also "chemotherapy").
- Skip syntactic words, articles, and recommendation-strength words (should, may, must).
- Return strictly JSON: {"entities":[{"surface_form":...,"normalized_form":...,"semantic_group":...}]}
- If no entities, return {"entities":[]}"""

ENTITY_FEW_SHOT_USER = '[SENTENCE]\n"Patient is a 65-year-old man with stage III non-small cell lung cancer and EGFR mutation."'

ENTITY_FEW_SHOT_ASSISTANT = json.dumps({
    "entities": [
        {"surface_form": "65-year-old", "normalized_form": "Aged", "semantic_group": "LIVB"},
        {"surface_form": "man", "normalized_form": "Male", "semantic_group": "LIVB"},
        {"surface_form": "stage III", "normalized_form": "Stage III", "semantic_group": "CONC"},
        {"surface_form": "non-small cell lung cancer", "normalized_form": "Non-Small Cell Lung Carcinoma", "semantic_group": "DISO"},
        {"surface_form": "lung cancer", "normalized_form": "Lung Neoplasm", "semantic_group": "DISO"},
        {"surface_form": "EGFR mutation", "normalized_form": "EGFR Gene Mutation", "semantic_group": "DISO"},
        {"surface_form": "EGFR", "normalized_form": "Epidermal Growth Factor Receptor", "semantic_group": "GENE"},
    ]
})


CONDITION_SYSTEM_PROMPT = """Extract structured conditions describing the patient or case from each clinical sentence. No triple anchor is given — extract any eligibility, clinical state, exposure, or temporal context the sentence asserts about the patient.

[CONDITION TYPES — pick exactly one per condition]

1. numeric_threshold — numeric cutoffs, ranges, labs, vital signs, age, scores, imaging measurements, duration cutoffs.
   Required: type, variable, comparator, value.
   Range form: comparator="between" with value_min and value_max (value may be a compact string like "55-74"). inclusive_min/inclusive_max default to true.
   Optional: unit, subtype, evidence_text, value_min, value_max, inclusive_min, inclusive_max.
   Allowed subtypes: age, vital_sign, lab_value, score, imaging, duration.
   Allowed comparators: <, <=, =, >=, >, between.

2. categorical_state — non-numeric diagnosis, state, risk, patient population, stage, severity, care setting, procedure/intervention history.
   Required: type, variable, value.
   Optional: subtype, evidence_text, clinical_status, verification_status, severity, body_site, value_operator.
   Allowed subtypes: patient_context, population, clinical_state, stage_severity, risk_status, care_setting, procedure_history, intervention_status.
   Status dimensions (use the dimension(s) supported by the sentence):
     clinical_status: active, inactive, stable, unstable, resolved, present, absent, current, past, no_history, unknown.
     verification_status: confirmed, suspected, excluded, contraindicated, unknown.
     severity: mild, moderate, severe.

3. medication_history — current/prior drug exposure, failure, discontinuation, contraindication, or dose/frequency/route.
   Required: type, drug, status.
   Optional: subtype, evidence_text, dose, unit, frequency, route.
   Allowed subtypes: current_medication, prior_medication, medication_failure, discontinued_medication, contraindicated_medication, dose_condition.
   Allowed status: current, prior, past, failed, discontinued, contraindicated, not_current, unknown.

4. temporal_condition — time windows, follow-up, onset timing, temporal order, frequency, duration with an anchor event.
   Required: type, event, anchor, comparator.
   Optional: subtype, evidence_text, interval, interval_unit, temporal_relation.
   Allowed subtypes: time_window, temporal_order, duration, follow_up, frequency.
   Allowed temporal_relation: before, after, during, overlaps, contains, equals, starts, finishes, meets.
   comparator handles numeric size of the interval (<= 15 years); temporal_relation handles qualitative ordering (before / after / during).

[NEGATIVE OR EXCLUSION CONDITIONS]
Encode negation via the dimensional fields, not by adding NOT:
  "without diabetes"               → categorical_state, clinical_status="absent"
  "no history of stroke"           → categorical_state, clinical_status="no_history"
  "contraindicated in pregnancy"   → categorical_state, verification_status="contraindicated"
  "discontinued metformin"         → medication_history, status="discontinued"
  "not currently taking aspirin"   → medication_history, status="not_current"

[PRIORITY — choose the highest-priority matching type]
1. medication_history (e.g. "prednisone ≥20 mg/day")
2. numeric_threshold  (e.g. "age 55-74", "Hb < 8 g/dL")
3. temporal_condition (e.g. "within 72 hours after onset")
4. categorical_state  (e.g. "severe active diabetes", "underwent appendectomy")

[RULES]
- Extract every applicable patient-describing condition. Skip incidental mentions unrelated to the patient/case state.
- Prefer a single numeric_threshold with comparator="between" + value_min/value_max for conceptual ranges (splitting into two >= / <= is also acceptable).
- evidence_text: ≤50 chars, verbatim phrase from the sentence. Required when possible; omit if not directly quotable.
- Do not include qualifies, condition_logic, or condition_source fields — they are irrelevant at query time.

[OUTPUT]
Return a JSON object {"results":[...]} with EXACTLY one entry per input sentence_index (0..N-1). If a sentence has no applicable condition, still include its entry with conditions:[]."""

CONDITION_FEW_SHOT_USER = """[SENTENCES]
Sentence 0: "Screening recommended for adults aged 55-74 with 30+ pack-years who quit within 15 years."
Sentence 1: "Patient has severe active rheumatoid arthritis but no history of stroke."
Sentence 2: "She started metformin 500 mg twice daily orally several weeks ago and failed first-line antibiotics last year."
Sentence 3: "Systolic blood pressure is 88 mmHg; vomiting persisted during the first 2 hours from presentation."
"""

CONDITION_FEW_SHOT_ASSISTANT = json.dumps({
    "results": [
        {"sentence_index": 0, "conditions": [
            {"type": "numeric_threshold", "subtype": "age", "variable": "age", "comparator": "between",
             "value": "55-74", "value_min": 55, "value_max": 74, "inclusive_min": True, "inclusive_max": True,
             "unit": "years", "evidence_text": "aged 55-74"},
            {"type": "numeric_threshold", "subtype": "duration", "variable": "smoking_pack_year",
             "comparator": ">=", "value": 30, "unit": "pack-years", "evidence_text": "30+ pack-years"},
            {"type": "temporal_condition", "subtype": "time_window", "event": "smoking_cessation",
             "anchor": "presentation", "comparator": "<=", "interval": 15, "interval_unit": "years",
             "temporal_relation": "before", "evidence_text": "quit within 15 years"},
        ]},
        {"sentence_index": 1, "conditions": [
            {"type": "categorical_state", "subtype": "clinical_state", "variable": "rheumatoid_arthritis",
             "value": "severe active rheumatoid arthritis", "clinical_status": "active", "severity": "severe",
             "evidence_text": "severe active rheumatoid arthritis"},
            {"type": "categorical_state", "subtype": "clinical_state", "variable": "stroke",
             "value": "no history of stroke", "clinical_status": "no_history",
             "evidence_text": "no history of stroke"},
        ]},
        {"sentence_index": 2, "conditions": [
            {"type": "medication_history", "subtype": "current_medication", "drug": "metformin",
             "status": "current", "dose": "500", "unit": "mg", "frequency": "twice daily", "route": "oral",
             "evidence_text": "metformin 500 mg twice daily"},
            {"type": "temporal_condition", "subtype": "time_window", "event": "metformin_initiation",
             "anchor": "presentation", "comparator": "=", "interval": 2, "interval_unit": "weeks",
             "temporal_relation": "before", "evidence_text": "several weeks ago"},
            {"type": "medication_history", "subtype": "medication_failure", "drug": "first-line antibiotics",
             "status": "failed", "evidence_text": "failed first-line antibiotics"},
        ]},
        {"sentence_index": 3, "conditions": [
            {"type": "numeric_threshold", "subtype": "vital_sign", "variable": "systolic_blood_pressure",
             "comparator": "=", "value": 88, "unit": "mmHg", "evidence_text": "systolic blood pressure is 88"},
            {"type": "temporal_condition", "subtype": "duration", "event": "vomiting", "anchor": "presentation",
             "comparator": "=", "interval": 2, "interval_unit": "hours", "temporal_relation": "during",
             "evidence_text": "during the first 2 hours"},
        ]},
    ]
})


ANSWER_SYSTEM_PROMPT = (
    "You are answering a medical multiple-choice question. "
    "Use the retrieved knowledge below as supporting evidence "
    "(it may be partially relevant). Pick exactly one option symbol "
    "such as A, B, C, or D. "
    'Output ONLY a JSON object: {"answer": "<option symbol>"}. '
)

BASELINE_SYSTEM_PROMPT = (
    "You are answering a medical multiple-choice question. "
    "Pick exactly one option symbol such as A, B, C, or D. "
    'Output ONLY a JSON object: {"answer": "<option symbol>"}. '
)


# ──────────────────────────────────────────────────────────────────────
# Extractors
# ──────────────────────────────────────────────────────────────────────

class LlamaEntityExtractor:
    """Per-sentence entity extraction via the local LLM."""

    def __init__(self, llm: BaseLLM, cfg: ExperimentConfig):
        self.llm = llm
        self.cfg = cfg

    def extract_per_sentence(self, sentences: list[str]) -> list[list[dict]]:
        if not sentences:
            return []
        batch = [
            [
                {"role": "system", "content": ENTITY_SYSTEM_PROMPT},
                {"role": "user", "content": ENTITY_FEW_SHOT_USER},
                {"role": "assistant", "content": ENTITY_FEW_SHOT_ASSISTANT},
                {"role": "user", "content": f'[SENTENCE]\n"{s}"'},
            ]
            for s in sentences
        ]
        outputs: list[list[dict]] = []
        bs = max(1, self.cfg.llm_batch_size)
        for i in range(0, len(batch), bs):
            chunk = batch[i : i + bs]
            raws = self.llm.generate_batch(
                chunk,
                max_new_tokens=self.cfg.llm_max_new_tokens_extraction,
                json_mode=True,
            )
            for raw in raws:
                outputs.append(_parse_entity_response(raw))
        return outputs

    @staticmethod
    def dedup_entities(per_sent: list[list[dict]]) -> list[dict]:
        seen, out = set(), []
        for ents in per_sent:
            for ent in ents:
                key = (
                    (ent.get("normalized_form") or ent.get("surface_form") or "")
                    .lower().strip()
                )
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(ent)
        return out


def _parse_entity_response(raw: str) -> list[dict]:
    data = parse_json_object(raw)
    if not isinstance(data, dict):
        return []
    ents = data.get("entities", [])
    if not isinstance(ents, list):
        return []
    # EntityMatcher only reads surface_form / normalized_form / semantic_group,
    # so we keep the dict minimal (TUI/name fields are vestigial here).
    out = []
    for e in ents:
        if not isinstance(e, dict):
            continue
        sf = e.get("surface_form")
        if not sf:
            continue
        out.append({
            "surface_form": str(sf),
            "normalized_form": str(e.get("normalized_form", sf) or sf),
            "semantic_group": str(e.get("semantic_group", "") or ""),
        })
    return out


class LlamaConditionExtractor:
    """Joint condition extraction across all sentences in a single call."""

    def __init__(self, llm: BaseLLM, cfg: ExperimentConfig):
        self.llm = llm
        self.cfg = cfg

    # One condition entry is roughly 80-120 output tokens; budget ~100 per sentence.
    _TOKENS_PER_SENTENCE = 100
    _SENTENCE_WARN_THRESHOLD = 10

    def extract_per_sentence(self, sentences: list[str]) -> list[list[dict]]:
        if not sentences:
            return []
        n = len(sentences)
        if n > self._SENTENCE_WARN_THRESHOLD:
            logger.warning(
                f"LlamaConditionExtractor: {n} sentences exceed the warning "
                f"threshold of {self._SENTENCE_WARN_THRESHOLD}. "
                f"JSON output may be truncated at max_new_tokens="
                f"{self.cfg.llm_max_new_tokens_extraction}."
            )
        max_new_tokens = max(
            self.cfg.llm_max_new_tokens_extraction,
            n * self._TOKENS_PER_SENTENCE,
        )
        user_lines = ["[SENTENCES]"]
        for i, s in enumerate(sentences):
            user_lines.append(f'Sentence {i}: "{s}"')
        messages = [
            {"role": "system", "content": CONDITION_SYSTEM_PROMPT},
            {"role": "user", "content": CONDITION_FEW_SHOT_USER},
            {"role": "assistant", "content": CONDITION_FEW_SHOT_ASSISTANT},
            {"role": "user", "content": "\n".join(user_lines)},
        ]
        raw = self.llm.generate(
            messages,
            max_new_tokens=max_new_tokens,
            json_mode=True,
        )
        data = parse_json_object(raw)
        per_sent: list[list[dict]] = [[] for _ in sentences]
        if not isinstance(data, dict):
            logger.warning(
                f"LlamaConditionExtractor: failed to parse JSON response "
                f"(possibly truncated). sentences={n}, "
                f"max_new_tokens={max_new_tokens}. "
                f"Returning empty conditions for all sentences."
            )
            return per_sent
        for r in data.get("results", []) or []:
            if not isinstance(r, dict):
                continue
            idx = r.get("sentence_index")
            if isinstance(idx, int) and 0 <= idx < len(sentences):
                for cond in r.get("conditions", []) or []:
                    if isinstance(cond, dict):
                        per_sent[idx].append(cond)
        return per_sent


def conditions_to_keywords(
    conditions: list[dict],
    min_len: int,
    max_count: int,
) -> list[str]:
    """Pull substring-matchable keywords from extracted conditions.

    The KG stores conditions as a JSON string in r.conditions_json, so we use
    case-insensitive substring matching for CYPHER_31 / CYPHER_33. Pulls every
    string-valued schema field that could appear verbatim in conditions_json:
      - identifiers / values: variable, value, drug, event, anchor, evidence_text
      - categorical_state dimensions: subtype, clinical_status, verification_status,
        severity, body_site
      - medication_history detail: status, dose, frequency, route
      - temporal_condition detail: temporal_relation, interval_unit
    Numeric-only fields (value_min, value_max, interval, inclusive_*) are excluded
    because raw numbers are noisy keywords.
    """
    seen: set[str] = set()
    out: list[str] = []
    fields = (
        # Core identifiers / surface text.
        "variable", "value", "drug", "event", "anchor", "evidence_text",
        # Controlled-vocab attributes that are stored as strings in conditions_json.
        "subtype",
        "clinical_status", "verification_status", "severity", "body_site",
        "status", "dose", "frequency", "route",
        "temporal_relation", "interval_unit",
    )
    for c in conditions:
        for f in fields:
            v = c.get(f)
            if v is None or isinstance(v, bool):
                continue
            s = str(v).strip().lower()
            if len(s) < min_len or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= max_count:
                return out
    return out


# ──────────────────────────────────────────────────────────────────────
# Answer prompt builders + parser
# ──────────────────────────────────────────────────────────────────────

def build_answer_user_prompt(retrieval_text: str, sample: Sample) -> str:
    options_block = "\n".join(f"{k}. {v}" for k, v in sample.options.items())
    return (
        f"[RETRIEVAL]\n{retrieval_text}\n\n"
        f"[QUESTION]\n{sample.question}\n\n"
        f"[OPTIONS]\n{options_block}"
    )


def build_baseline_user_prompt(sample: Sample) -> str:
    """Variant 1: no retrieval block. Mirrors mediq_graphrag_test.py baseline."""
    options_block = "\n".join(f"{k}. {v}" for k, v in sample.options.items())
    return f"[QUESTION]\n{sample.question}\n\n[OPTIONS]\n{options_block}"


def parse_answer(raw: str, options: dict[str, str]) -> str:
    data = parse_json_object(raw)
    if isinstance(data, dict):
        ans = str(data.get("answer", "")).strip().upper()
        if ans:
            # Exact key match: "A", "B", "C", "D"
            if ans in options:
                return ans
            # "D. Syphilis" or "D) Syphilis" → extract leading key letter
            m = re.match(r'^([A-Z])[.\)]\s*', ans)
            if m and m.group(1) in options:
                return m.group(1)
            # Full text match (case-insensitive)
            for k, v in options.items():
                if v and v.lower() == ans.lower():
                    return k
            return ans
    if raw:
        low = raw.lower()
        for k, v in options.items():
            if k.lower() == low.strip():
                return k
        for v in options.values():
            if v and v.lower() in low:
                for k, option_text in options.items():
                    if option_text == v:
                        return k
    return ""


def normalize_for_compare(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# ──────────────────────────────────────────────────────────────────────
# Variant runners
# ──────────────────────────────────────────────────────────────────────

class BaseRunner(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, sample: Sample) -> dict: ...


class OnlyLLMRunner(BaseRunner):
    """Variant 1: question + options, no retrieval."""
    name = "only_llm"

    def __init__(self, llm: BaseLLM, cfg: ExperimentConfig):
        self.llm = llm
        self.cfg = cfg

    def run(self, sample: Sample) -> dict:
        t0 = time.perf_counter()
        user_prompt = build_baseline_user_prompt(sample)
        messages = [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.llm.generate(
            messages,
            max_new_tokens=self.cfg.llm_max_new_tokens_answer,
            json_mode=True,
        )
        predicted = parse_answer(raw, sample.options)
        elapsed = time.perf_counter() - t0
        token_counts = _count_tokens(self.llm, retrieval=None,
                                     messages=messages, raw=raw)
        return _record(
            sample=sample,
            predicted=predicted,
            raw=raw,
            elapsed=elapsed,
            user_prompt=user_prompt,
            system_prompt=BASELINE_SYSTEM_PROMPT,
            retrieval=None,
            token_counts=token_counts,
        )


class VectorRAGRunner(BaseRunner):
    """Variant 2: top-k cosine over Stage-0 recommendation embeddings."""
    name = "vector_rag"

    def __init__(self, llm: BaseLLM, cfg: ExperimentConfig, retriever: VectorRAGRetriever):
        self.llm = llm
        self.cfg = cfg
        self.retriever = retriever

    def run(self, sample: Sample) -> dict:
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(sample.input_query())
        user_prompt = build_answer_user_prompt(retrieval.formatted_text, sample)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.llm.generate(
            messages,
            max_new_tokens=self.cfg.llm_max_new_tokens_answer,
            json_mode=True,
        )
        predicted = parse_answer(raw, sample.options)
        elapsed = time.perf_counter() - t0
        token_counts = _count_tokens(self.llm, retrieval=retrieval,
                                     messages=messages, raw=raw)
        return _record(
            sample=sample,
            predicted=predicted,
            raw=raw,
            elapsed=elapsed,
            user_prompt=user_prompt,
            system_prompt=ANSWER_SYSTEM_PROMPT,
            retrieval=retrieval,
            token_counts=token_counts,
        )


class KGNoCondRunner(BaseRunner):
    """Variant 3: entity extraction → UMLS CUI → 1-hop subgraph + 2-hop paths."""
    name = "kg_no_cond"

    def __init__(
        self,
        llm: BaseLLM,
        cfg: ExperimentConfig,
        entity_extractor: LlamaEntityExtractor,
        umls: CachedUMLSMatcher,
        retriever: KGNoConditionsRetriever,
    ):
        self.llm = llm
        self.cfg = cfg
        self.entity_extractor = entity_extractor
        self.umls = umls
        self.retriever = retriever

    def run(self, sample: Sample) -> dict:
        t0 = time.perf_counter()
        sentences = split_sentences(sample.input_query())
        per_sent_entities = self.entity_extractor.extract_per_sentence(sentences)
        entities = LlamaEntityExtractor.dedup_entities(per_sent_entities)
        _, cuis = self.umls.match_many(entities)
        retrieval = self.retriever.retrieve(cuis)

        user_prompt = build_answer_user_prompt(retrieval.formatted_text, sample)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.llm.generate(
            messages,
            max_new_tokens=self.cfg.llm_max_new_tokens_answer,
            json_mode=True,
        )
        predicted = parse_answer(raw, sample.options)
        elapsed = time.perf_counter() - t0
        token_counts = _count_tokens(self.llm, retrieval=retrieval,
                                     messages=messages, raw=raw)
        rec = _record(
            sample=sample,
            predicted=predicted,
            raw=raw,
            elapsed=elapsed,
            user_prompt=user_prompt,
            system_prompt=ANSWER_SYSTEM_PROMPT,
            retrieval=retrieval,
            token_counts=token_counts,
        )
        rec.update({
            "n_sentences": len(sentences),
            "n_entities": len(entities),
            "matched_cuis": cuis,
            "n_matched_cuis": len(cuis),
        })
        return rec


class KGWithCondRunner(BaseRunner):
    """Variant 4: entity + condition extraction → 3-1 / 3-2 / 3-3 cascade."""
    name = "kg_with_cond"

    def __init__(
        self,
        llm: BaseLLM,
        cfg: ExperimentConfig,
        entity_extractor: LlamaEntityExtractor,
        condition_extractor: LlamaConditionExtractor,
        umls: CachedUMLSMatcher,
        retriever: KGWithConditionsRetriever,
    ):
        self.llm = llm
        self.cfg = cfg
        self.entity_extractor = entity_extractor
        self.condition_extractor = condition_extractor
        self.umls = umls
        self.retriever = retriever

    def run(self, sample: Sample) -> dict:
        t0 = time.perf_counter()
        sentences = split_sentences(sample.input_query())
        per_sent_entities = self.entity_extractor.extract_per_sentence(sentences)
        entities = LlamaEntityExtractor.dedup_entities(per_sent_entities)
        _, cuis = self.umls.match_many(entities)

        per_sent_conditions = self.condition_extractor.extract_per_sentence(sentences)
        all_conditions: list[dict] = []
        for conds in per_sent_conditions:
            all_conditions.extend(conds)
        cond_kws = conditions_to_keywords(
            all_conditions,
            min_len=self.cfg.cond_keyword_min_len,
            max_count=self.cfg.cond_keyword_max,
        )

        retrieval = self.retriever.retrieve(cuis, cond_kws)
        user_prompt = build_answer_user_prompt(retrieval.formatted_text, sample)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.llm.generate(
            messages,
            max_new_tokens=self.cfg.llm_max_new_tokens_answer,
            json_mode=True,
        )
        predicted = parse_answer(raw, sample.options)
        elapsed = time.perf_counter() - t0
        token_counts = _count_tokens(self.llm, retrieval=retrieval,
                                     messages=messages, raw=raw)
        rec = _record(
            sample=sample,
            predicted=predicted,
            raw=raw,
            elapsed=elapsed,
            user_prompt=user_prompt,
            system_prompt=ANSWER_SYSTEM_PROMPT,
            retrieval=retrieval,
            token_counts=token_counts,
        )
        rec.update({
            "n_sentences": len(sentences),
            "n_entities": len(entities),
            "matched_cuis": cuis,
            "n_matched_cuis": len(cuis),
            "n_conditions": len(all_conditions),
            "condition_keywords": cond_kws,
        })
        return rec


# ──────────────────────────────────────────────────────────────────────
# Token counting + record builder
# ──────────────────────────────────────────────────────────────────────

def _count_tokens(
    llm: BaseLLM,
    retrieval: Optional[RetrievalResult],
    messages: list[dict],
    raw: str,
) -> dict:
    """Per-sample answer-generation token accounting.

    - retrieval_tokens: tokens of the retrieval block text alone (0 for only_llm).
    - answer_input_tokens: chat-templated prompt length actually fed to .generate
      (system + retrieval + question + options + Llama chat-template framing).
    - answer_output_tokens: tokens of the model's raw response.
    - answer_total_tokens: input + output (one full forward+decode pass).
    """
    # count_tokens / count_prompt_tokens are LocalLLM-specific; gracefully fall
    # back to 0 for backends that don't expose them (e.g. future API backends).
    if not (hasattr(llm, "count_tokens") and hasattr(llm, "count_prompt_tokens")):
        return {
            "retrieval_tokens": 0,
            "answer_input_tokens": 0,
            "answer_output_tokens": 0,
            "answer_total_tokens": 0,
        }
    retrieval_tokens = (
        llm.count_tokens(retrieval.formatted_text) if retrieval else 0
    )
    input_tokens = llm.count_prompt_tokens(messages, json_mode=True)
    output_tokens = llm.count_tokens(raw)
    return {
        "retrieval_tokens": retrieval_tokens,
        "answer_input_tokens": input_tokens,
        "answer_output_tokens": output_tokens,
        "answer_total_tokens": input_tokens + output_tokens,
    }


def _record(
    sample: Sample,
    predicted: str,
    raw: str,
    elapsed: float,
    user_prompt: str,
    system_prompt: str,
    retrieval: Optional[RetrievalResult],
    token_counts: dict,
) -> dict:
    correct = normalize_for_compare(predicted) == normalize_for_compare(sample.gold_answer_idx)
    return {
        "predicted": predicted,
        "correct": bool(correct),
        "elapsed_seconds": round(elapsed, 4),
        "raw_response": raw,
        "user_prompt": user_prompt,
        "system_prompt": system_prompt,
        "retrieval_strategy": retrieval.strategy if retrieval else "no_retrieval",
        "retrieval_n_items": retrieval.n_items if retrieval else 0,
        "retrieval_text": retrieval.formatted_text if retrieval else "",
        "retrieval_debug": retrieval.debug if retrieval else {},
        # Token accounting (Llama tokenizer; matches what model.generate saw).
        "retrieval_tokens": token_counts.get("retrieval_tokens", 0),
        "answer_input_tokens": token_counts.get("answer_input_tokens", 0),
        "answer_output_tokens": token_counts.get("answer_output_tokens", 0),
        "answer_total_tokens": token_counts.get("answer_total_tokens", 0),
    }
