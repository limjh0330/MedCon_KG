"""
Stage 3: Condition Augmentation

Attaches structured conditions to UMLS layer triples by comparing each triple
against its source CREST guideline recommendation(s).

Matching strategy:
  Triple head/tail names are UMLS preferred names while entity candidates use
  LLM-normalized forms. Direct string matching fails, so we use CUI-based
  matching via stage2_umls_matched.json:
    triple.head_cui → match_results → entity.source_guidelines → recommendations

  Bidirectional filter: a triple is sent to the LLM only when head AND tail
  both resolve to a shared recommendation. Triples without such a rec are
  passed through with empty conditions (has_conditions=False)

Condition types (4):
  numeric_threshold, categorical_state, medication_history, temporal_condition
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import config

logger = logging.getLogger(__name__)

_openai_client = None

# UMLS CUI: literal "C" + 7 digits. Tighter than startswith("C") so we don't
# misclassify source codes that happen to start with "C" (e.g. HCPCS "C1300").
_CUI_RE = re.compile(r"^C\d{7}$")


def _get_openai_client(api_key: str = None):
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
    return _openai_client


# ──────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────

CONDITION_SYSTEM_PROMPT = """Extract structured conditions from guideline recommendation sentences that constrain medical triples.

You will receive:
1. Medical triples: (head_name, relation, tail_name)
2. Guideline recommendation sentence(s) aligned with each triple

Your task:
For each input triple, extract only the conditions that actually constrain the validity, applicability, or interpretation of that triple or its aligned recommendation/action sentence. Do not extract unrelated mentions.

[CORE PRINCIPLE]
- Entities are the head or tail of the primary triple.
- Conditions are qualifier values: they restrict when, for whom, under what clinical state, history/exposure, numeric criterion, or temporal context the triple or recommendation is valid.
- Pick exactly one condition type per condition.
- Keep required fields minimal. Add optional fields only when clearly supported by the text.

[CONDITION TYPES]

1. numeric_threshold
Use for numeric cutoffs, numeric ranges, laboratory values, vital signs, age, scores, imaging measurements, physiologic measurements, or duration cutoffs.
Required fields: type, variable, comparator, value.
Range exception: for a conceptual range, use comparator="between" with value_min and value_max. In that case value may be a compact string such as "55-74" or may be omitted if validation allows range fields.
Optional fields: unit, evidence_text, subtype, qualifies, value_min, value_max, inclusive_min, inclusive_max.
Allowed subtypes: age, vital_sign, lab_value, score, imaging, duration.

Examples:
- "systolic blood pressure <90 mmHg"
  → {"type":"numeric_threshold","subtype":"vital_sign","variable":"systolic_blood_pressure","comparator":"<","value":90,"unit":"mmHg","evidence_text":"systolic blood pressure <90 mmHg","qualifies":"relation"}
- "adults aged 55-74"
  → {"type":"numeric_threshold","subtype":"age","variable":"age","comparator":"between","value":"55-74","value_min":55,"value_max":74,"inclusive_min":true,"inclusive_max":true,"unit":"years","evidence_text":"aged 55-74","qualifies":"recommendation"}

2. categorical_state
Use for non-numeric diagnosis/state/risk/patient population/stage/severity/care setting/procedure/intervention context, or when the condition modifies the head or tail entity.
Required fields: type, variable, value.
Optional fields: evidence_text, subtype, qualifies, clinical_status, verification_status, severity, body_site, value_operator.
Allowed subtypes: patient_context, population, clinical_state, stage_severity, risk_status, care_setting, head_modifier, tail_modifier, procedure_history, intervention_status.

Status dimensions for categorical_state:
- clinical_status: active, inactive, stable, unstable, resolved, present, absent, current, past, no_history, unknown.
- verification_status: confirmed, suspected, excluded, contraindicated, unknown.
- severity: mild, moderate, severe.
Do not use the old single status field for categorical_state unless unavoidable; prefer clinical_status, verification_status, and severity.

Examples:
- "stable blunt trauma patients"
  → {"type":"categorical_state","subtype":"patient_context","variable":"blunt_trauma","value":"stable blunt trauma patients","clinical_status":"stable","evidence_text":"stable blunt trauma patients","qualifies":"relation"}
- "gross hematuria or microscopic hematuria"
  → {"type":"categorical_state","subtype":"tail_modifier","variable":"hematuria","value":"gross or microscopic hematuria","value_operator":"OR","evidence_text":"gross hematuria or microscopic hematuria","qualifies":"tail"}
- "patients who underwent appendectomy"
  → {"type":"categorical_state","subtype":"procedure_history","variable":"appendectomy","value":"underwent appendectomy","clinical_status":"past","evidence_text":"underwent appendectomy","qualifies":"relation"}
- "severe active diabetes"
  → {"type":"categorical_state","subtype":"clinical_state","variable":"diabetes","value":"severe active diabetes","clinical_status":"active","severity":"severe","evidence_text":"severe active diabetes","qualifies":"relation"}

3. medication_history
Use for current or prior medication exposure, medication failure, discontinued medication, contraindicated medication, or medication dose/frequency/route condition.
Required fields: type, drug, status.
Optional fields: evidence_text, subtype, dose, unit, frequency, route, qualifies.
Allowed subtypes: current_medication, prior_medication, medication_failure, discontinued_medication, contraindicated_medication, dose_condition.
Allowed status values: current, prior, past, failed, discontinued, contraindicated, not_current, unknown.

Examples:
- "failed first-line antibiotics"
  → {"type":"medication_history","subtype":"medication_failure","drug":"first-line antibiotics","status":"failed","evidence_text":"failed first-line antibiotics","qualifies":"recommendation"}
- "metformin 500mg twice daily orally"
  → {"type":"medication_history","subtype":"current_medication","drug":"metformin","status":"current","dose":"500","unit":"mg","frequency":"twice daily","route":"oral","evidence_text":"metformin 500mg twice daily orally","qualifies":"relation"}

4. temporal_condition
Use for time windows, follow-up, onset timing, temporal order, frequency, duration, or event-anchor-based temporal constraints.
Required fields: type, event, anchor, comparator.
Optional fields: evidence_text, subtype, interval, interval_unit, temporal_relation, qualifies.
Allowed subtypes: time_window, temporal_order, duration, follow_up, frequency.
Allowed temporal_relation values: before, after, during, overlaps, contains, equals, starts, finishes, meets.

Important distinction:
- comparator handles arithmetic comparison of the interval or duration, such as <= 15 years.
- temporal_relation handles qualitative temporal relation between event/interval and anchor, such as before, after, during, overlaps.

Examples:
- "quit within 15 years before presentation"
  → {"type":"temporal_condition","subtype":"time_window","event":"smoking_cessation","anchor":"presentation","comparator":"<=","interval":15,"interval_unit":"years","temporal_relation":"before","evidence_text":"quit within 15 years","qualifies":"recommendation"}
- "vomiting during the first 2 hours from presentation"
  → {"type":"temporal_condition","subtype":"duration","event":"vomiting","anchor":"presentation","comparator":"=","interval":2,"interval_unit":"hours","temporal_relation":"during","evidence_text":"vomiting for 2 hours from presentation","qualifies":"relation"}

[OPTIONAL FIELDS — include only when clear]
- subtype: use only the controlled subtype values listed above.
- qualifies: one of relation, head, tail, recommendation. Use relation by default.
  If the condition refines the head entity, use head.
  If the condition refines the tail entity, use tail.
  If the condition applies to the whole guideline recommendation/action, use recommendation.
- value_operator: use only when the condition value itself contains internal alternatives or conjunctions. Allowed values: AND, OR. Do NOT use NOT as value_operator.
- body_site: use for anatomical/procedural/lesion site when explicitly stated.
- frequency and route: use only for medication_history.
- temporal_relation: use only for temporal_condition qualitative temporal relations.
- value_min/value_max/inclusive_min/inclusive_max: use only for numeric_threshold ranges.

[NEGATIVE OR EXCLUSION CONDITIONS]
- Do not use NOT as value_operator.
- For categorical_state negative/exclusion/no-history/contraindication conditions, use dimensional fields:
  "without diabetes" → clinical_status="absent"
  "no history of stroke" → clinical_status="no_history"
  "exclude patients with renal failure" → verification_status="excluded"
  "contraindicated in pregnancy" → verification_status="contraindicated"
- For medication_history, use the medication status field:
  "discontinued metformin" → status="discontinued"
  "not currently taking anticoagulants" → status="not_current"
  "contraindicated NSAIDs" → status="contraindicated"
- Use condition_logic="NOT" only when the entire condition set is explicitly negated.

[ENTITY-CONDITION SEPARATION RULES]
- Do not duplicate a head or tail entity as a condition unless the condition adds subtype, body site, severity, clinical/verification status, numeric threshold, temporal information, exposure history, procedure/intervention status, or contextual restriction.
- If a condition only repeats the head or tail without adding any extra restriction, do not extract it.
- If the condition refines the head or tail entity, set qualifies to head or tail.
- If the condition restricts applicability of the relation or recommendation, set qualifies to relation or recommendation.

[PRIORITY RULE]
If a phrase appears to fit multiple condition types, choose the type using this priority:
1. medication_history
2. numeric_threshold
3. temporal_condition
4. categorical_state

Examples:
- "prednisone ≥20 mg/day" → medication_history, because it is a medication dose condition.
- "age ≥65 years" or "age 55-74" → numeric_threshold.
- "within 72 hours after onset" → temporal_condition.
- "older adult" or "severe active diabetes" → categorical_state.
- "underwent appendectomy" → categorical_state with subtype procedure_history.

[RULES]
- Only attach conditions that actually constrain the triple relation or the aligned recommendation/action; ignore unrelated mentions in the rec.
- For numeric ranges, prefer a single numeric_threshold with comparator="between", value_min, and value_max when the phrase is conceptually one range. Splitting into two numeric_threshold conditions remains allowed for backward compatibility.
- Each condition should include evidence_text (≤50 chars, verbatim phrase from the rec). If uncertain, omit it; validation will not fail solely for missing evidence_text.
- condition_logic must be one of "AND" | "OR" | "NOT" (use "AND" when only one condition).
- condition_source: {"guideline_id", "evidence_level", "evidence_texts":[...]} — evidence_level is "sentence_aligned" when the condition is in the same recommendation sentence, "guideline_cooccurrence" when only co-located in the guideline, or "inferred".

[OUTPUT]
Return a JSON object {"results":[...]} with EXACTLY one entry per input triple_index (0..N-1). If a triple has no applicable condition, still include its entry with conditions:[]."""

FEW_SHOT_USER = """[TRIPLES]
Triple 0:
  ("Lung Neoplasms", "screened_by", "Low-dose CT")
Triple 1:
  ("Lung Neoplasms", "is_a", "Neoplasm")

[RECOMMENDATION SENTENCES]
Rec for Triple 0:
  "screening recommended for adults aged 55-74, 30+ pack-years, quit within 15 years"
  [guideline_id: g42, strength: strong]
Rec for Triple 1:
  "screening recommended for adults aged 55-74, 30+ pack-years, quit within 15 years"
  [guideline_id: g42, strength: strong]"""

FEW_SHOT_ASSISTANT = """{"results":[{"triple_index":0,"conditions":[{"type":"numeric_threshold","subtype":"age","variable":"age","comparator":"between","value":"55-74","value_min":55,"value_max":74,"inclusive_min":true,"inclusive_max":true,"unit":"years","evidence_text":"aged 55-74","qualifies":"recommendation"},{"type":"numeric_threshold","subtype":"duration","variable":"smoking_pack_year","comparator":">=","value":30,"unit":"pack-years","evidence_text":"30+ pack-years","qualifies":"recommendation"},{"type":"temporal_condition","subtype":"time_window","event":"smoking_cessation","anchor":"presentation","interval":15,"interval_unit":"years","comparator":"<=","temporal_relation":"before","evidence_text":"quit within 15 years","qualifies":"recommendation"}],"condition_logic":"AND","condition_source":{"guideline_id":"g42","evidence_level":"sentence_aligned","evidence_texts":["aged 55-74","30+ pack-years","quit within 15 years"]}},{"triple_index":1,"conditions":[]}]}"""



# ──────────────────────────────────────────────────────────────────
# CUI-based Recommendation Matching
# ──────────────────────────────────────────────────────────────────

def build_recommendation_index(
    recommendations: list[dict],
    entities: list[dict],
    match_results: list[dict],
) -> dict:
    """Build CUI→rec-indices and name→rec-indices lookup tables."""
    guideline_to_rec_idx: dict[str, list[int]] = {}
    for i, rec in enumerate(recommendations):
        gid = rec.get("guideline_id", "")
        if gid:
            guideline_to_rec_idx.setdefault(gid, []).append(i)

    cui_to_recs: dict[str, set] = {}
    for mr in match_results:
        if not mr.get("matched"):
            continue

        entity = mr.get("entity", {})
        rec_indices: set = set()
        for gid in entity.get("source_guidelines", []):
            for idx in guideline_to_rec_idx.get(gid, []):
                rec_indices.add(idx)

        if not rec_indices:
            continue

        for m in mr.get("matches", []):
            cui = m.get("cui", "")
            if _CUI_RE.match(cui):
                cui_to_recs.setdefault(cui, set()).update(rec_indices)

    # Name-based fallback for triples whose tail_id is not a CUI.
    unique_names: set[str] = set()
    for ent in entities:
        for k in ("normalized_form", "surface_form"):
            n = ent.get(k, "").lower().strip()
            if n and len(n) > 3:
                unique_names.add(n)

    rec_texts_lower = [rec.get("text", "").lower() for rec in recommendations]
    name_to_recs: dict[str, set] = {}
    for name in unique_names:
        hits = {i for i, t in enumerate(rec_texts_lower) if name in t}
        if hits:
            name_to_recs[name] = hits

    logger.info(
        f"Built recommendation index: {len(cui_to_recs)} CUIs mapped, "
        f"{len(name_to_recs)} name-based entries (fallback)"
    )

    return {"cui_to_recs": cui_to_recs, "name_to_recs": name_to_recs}


def find_relevant_recommendations(
    triple: dict,
    recommendations: list[dict],
    rec_index: dict,
    max_recs: int = None,
) -> list[dict]:
    """Recommendations mentioning BOTH endpoints. Empty list = skip LLM."""
    max_recs = max_recs or config.STAGE3_MAX_RECS_PER_TRIPLE

    cui_index = rec_index.get("cui_to_recs", {})
    name_index = rec_index.get("name_to_recs", {})

    head_cui = triple.get("head_cui", "")
    tail_id = triple.get("tail_id", "")
    head_name = triple.get("head_name", "").lower().strip()
    tail_name = triple.get("tail_name", "").lower().strip()

    head_recs = cui_index.get(head_cui, set())
    tail_recs = cui_index.get(tail_id, set()) if _CUI_RE.match(tail_id) else set()

    if not head_recs:
        head_recs = name_index.get(head_name, set())
    if not tail_recs:
        tail_recs = name_index.get(tail_name, set())

    both = head_recs & tail_recs
    if not both:
        return []

    # Sort for deterministic recs[0] (used for recommendation_strength).
    selected = sorted(both)[:max_recs]
    return [recommendations[i] for i in selected]


# ──────────────────────────────────────────────────────────────────
# LLM Call
# ──────────────────────────────────────────────────────────────────

def _build_user_message(
    triples_batch: list[dict],
    batch_recommendations: list[list[dict]],
) -> str:
    parts = ["[TRIPLES]"]
    for i, triple in enumerate(triples_batch):
        parts.append(
            f'Triple {i}:\n'
            f'  ("{triple["head_name"]}", "{triple["relation"]}", "{triple["tail_name"]}")'
        )

    parts.append("\n[RECOMMENDATION SENTENCES]")
    for i, recs in enumerate(batch_recommendations):
        for rec in recs:
            text = rec.get("text", "")
            if len(text) > config.STAGE3_REC_TEXT_MAX_CHARS:
                text = text[:config.STAGE3_REC_TEXT_MAX_CHARS].rsplit(" ", 1)[0] + "..."
            parts.append(
                f"Rec for Triple {i}:\n"
                f'  "{text}"\n'
                f"  [guideline_id: {rec.get('guideline_id', '')}, "
                f"strength: {rec.get('strength', '')}]"
            )

    return "\n".join(parts)


def _parse_condition_response(response_text: str) -> list[dict]:
    """Parse `{"results": [...]}` shape with a small truncation safety net."""
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Tier-1 salvage: trim to last `}` and synthesize closers. With
        # response_format=json_object we rarely reach this path, but if we do
        # the model has likely closed at least one object cleanly.
        last_brace = text.rfind("}")
        if last_brace != -1:
            for closer in ("]}", "}"):
                try:
                    data = json.loads(text[: last_brace + 1] + closer)
                    logger.warning("Recovered truncated JSON via brace-trim salvage")
                    break
                except json.JSONDecodeError:
                    continue

    if data is None:
        logger.warning(f"Failed to parse condition response: {text[:200]}")
        return []

    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            return results
        if isinstance(results, dict):
            return [results]
        return [data]
    if isinstance(data, list):
        return data
    return []


def extract_conditions_batch(
    triples_batch: list[dict],
    batch_recommendations: list[list[dict]],
    api_key: str = None,
    model: str = None,
    _retry_depth: int = 0,
) -> list[dict]:
    """Call LLM to extract conditions. Recursively retries only the missing
    triple_indices in small chunks so a partial failure doesn't trigger a full
    re-call."""
    client = _get_openai_client(api_key)
    model = model or config.LLM_MODEL

    messages = [
        {"role": "system", "content": CONDITION_SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": _build_user_message(triples_batch, batch_recommendations)},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            reasoning_effort=config.LLM_REASONING_EFFORT,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.error(f"Condition extraction LLM call failed: {e}")
        return []

    finish_reason = response.choices[0].finish_reason
    data = _parse_condition_response(response.choices[0].message.content)

    expected = set(range(len(triples_batch)))
    seen = {r.get("triple_index", -1) for r in data if isinstance(r, dict)}
    missing = sorted(expected - seen)
    if not missing:
        return data

    # Clean stop with at least one parsed result → model intentionally omitted
    # triples it judged to have no applicable condition. Fill with empty entries
    # rather than retry (the model would just omit them again).
    if finish_reason == "stop" and data:
        for mi in missing:
            data.append({"triple_index": mi, "conditions": []})
        return data

    if _retry_depth >= config.STAGE3_MAX_RETRY_DEPTH:
        logger.warning(
            f"Missing {len(missing)} result(s) after max retries; keeping partial"
        )
        return data

    logger.warning(
        f"Retrying {len(missing)}/{len(triples_batch)} missing triple(s) "
        f"at depth {_retry_depth + 1}"
    )

    for chunk_start in range(0, len(missing), config.STAGE3_RETRY_CHUNK_SIZE):
        chunk = missing[chunk_start: chunk_start + config.STAGE3_RETRY_CHUNK_SIZE]
        sub_triples = [triples_batch[i] for i in chunk]
        sub_recs = [batch_recommendations[i] for i in chunk]
        retry_data = extract_conditions_batch(
            sub_triples, sub_recs,
            api_key=api_key, model=model,
            _retry_depth=_retry_depth + 1,
        )
        for r in retry_data:
            if not isinstance(r, dict):
                continue
            local_idx = r.get("triple_index", -1)
            if isinstance(local_idx, int) and 0 <= local_idx < len(chunk):
                r["triple_index"] = chunk[local_idx]
                data.append(r)

    # Latest result per triple_index wins.
    merged_by_idx: dict = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        idx = item.get("triple_index", -1)
        if isinstance(idx, int):
            merged_by_idx[idx] = item
    return list(merged_by_idx.values())


# ──────────────────────────────────────────────────────────────────
# Condition Validation
# ──────────────────────────────────────────────────────────────────

VALID_CONDITION_TYPES = {
    "numeric_threshold", "categorical_state",
    "medication_history", "temporal_condition",
}

# Keep hard validation minimal to avoid dropping useful LLM outputs.
# numeric_threshold allows either the original single-value representation
# or the revised range representation using value_min/value_max.
REQUIRED_FIELDS = {
    "numeric_threshold": {"variable", "comparator"},
    "categorical_state": {"variable", "value"},
    "medication_history": {"drug", "status"},
    "temporal_condition": {"event", "anchor", "comparator"},
}

# Optional fields are controlled to keep schema drift and token usage bounded.
OPTIONAL_ALLOWED_FIELDS = {
    "evidence_text", "subtype", "qualifies", "value_operator",
    # Existing optional fields supported by the original prompt/schema.
    "unit", "dose", "interval", "interval_unit",
    # Revised numeric_threshold range representation.
    "value", "value_min", "value_max", "inclusive_min", "inclusive_max",
    # Revised categorical_state fields.
    "body_site", "clinical_status", "verification_status", "severity",
    # Backward-compatible old categorical_state status. It is mapped below.
    "status",
    # Revised medication_history dosage detail.
    "frequency", "route",
    # Revised temporal_condition qualitative temporal relation.
    "temporal_relation",
}

_ALLOWED_QUALIFIES = {"relation", "head", "tail", "recommendation"}
_ALLOWED_VALUE_OPERATORS = {"AND", "OR"}
_ALLOWED_COMPARATORS = {"<", "<=", "=", ">=", ">", "between"}

# medication_history.status keeps its own meaning and remains required.
_ALLOWED_MEDICATION_STATUS = {
    "current", "prior", "past", "failed", "discontinued",
    "contraindicated", "not_current", "unknown",
}

# categorical_state status is decomposed into three independent dimensions.
_ALLOWED_CLINICAL_STATUS = {
    "active", "inactive", "stable", "unstable", "resolved",
    "present", "absent", "current", "past", "no_history", "unknown",
}
_ALLOWED_VERIFICATION_STATUS = {
    "confirmed", "suspected", "excluded", "contraindicated", "unknown",
}
_ALLOWED_SEVERITY = {"mild", "moderate", "severe"}
_ALLOWED_TEMPORAL_RELATIONS = {
    "before", "after", "during", "overlaps", "contains",
    "equals", "starts", "finishes", "meets",
}

# Backward-compatible mapping from the old single categorical_state.status field.
_STATUS_TO_CATEGORICAL_FIELD = {
    "active": "clinical_status",
    "inactive": "clinical_status",
    "stable": "clinical_status",
    "unstable": "clinical_status",
    "resolved": "clinical_status",
    "present": "clinical_status",
    "absent": "clinical_status",
    "current": "clinical_status",
    "past": "clinical_status",
    "no_history": "clinical_status",
    "confirmed": "verification_status",
    "suspected": "verification_status",
    "excluded": "verification_status",
    "contraindicated": "verification_status",
    "mild": "severity",
    "moderate": "severity",
    "severe": "severity",
    "unknown": "clinical_status",
}

_ALLOWED_SUBTYPES = {
    "numeric_threshold": {"", "age", "vital_sign", "lab_value", "score", "imaging", "duration"},
    "categorical_state": {
        "", "patient_context", "population", "clinical_state", "stage_severity",
        "risk_status", "care_setting", "head_modifier", "tail_modifier",
        "procedure_history", "intervention_status",
    },
    "medication_history": {
        "", "current_medication", "prior_medication", "medication_failure",
        "discontinued_medication", "contraindicated_medication", "dose_condition",
    },
    "temporal_condition": {"", "time_window", "temporal_order", "duration", "follow_up", "frequency"},
}


def _validate_condition(cond: dict) -> bool:
    ctype = cond.get("type", "")
    if ctype not in VALID_CONDITION_TYPES:
        return False

    # numeric_threshold keeps the original fields for backward compatibility,
    # but accepts the revised range representation when comparator="between".
    if ctype == "numeric_threshold":
        if cond.get("variable") is None or cond.get("comparator") is None:
            return False
        has_single_value = cond.get("value") is not None
        has_range = cond.get("value_min") is not None and cond.get("value_max") is not None
        return has_single_value or has_range

    return all(cond.get(f) is not None for f in REQUIRED_FIELDS.get(ctype, set()))


def _coerce_bool(value, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return default


def _clean_optional_fields(cond: dict) -> dict:
    """Normalize the revised optional schema without making optional fields hard-required."""
    ctype = cond.get("type", "")

    # evidence_text is prompt-level requested but not hard validation-required.
    cond.setdefault("evidence_text", "")

    # Drop unapproved extra keys. Keep required fields and controlled optional fields.
    required = {"type"} | REQUIRED_FIELDS.get(ctype, set())
    if ctype == "numeric_threshold":
        required = {"type", "variable", "comparator"}
    for key in list(cond.keys()):
        if key not in required and key not in OPTIONAL_ALLOWED_FIELDS:
            cond.pop(key, None)

    # qualifies defaults to relation; invalid values are coerced rather than dropped.
    q = str(cond.get("qualifies", "relation")).strip().lower()
    cond["qualifies"] = q if q in _ALLOWED_QUALIFIES else "relation"

    # subtype is optional. Keep only compact controlled values to reduce sparse labels.
    subtype = str(cond.get("subtype", "")).strip().lower()
    subtype = subtype.replace("-", "_").replace(" ", "_")
    cond["subtype"] = subtype if subtype in _ALLOWED_SUBTYPES.get(ctype, {""}) else ""

    # comparator controlled normalization for numeric and temporal conditions.
    if ctype in {"numeric_threshold", "temporal_condition"} and "comparator" in cond:
        comp = str(cond.get("comparator", "")).strip().lower()
        if comp in {"≤", "=<"}:
            comp = "<="
        elif comp in {"≥", "=>"}:
            comp = ">="
        if comp not in _ALLOWED_COMPARATORS:
            # Preserve temporal qualitative comparators as temporal_relation when possible.
            if ctype == "temporal_condition" and comp in _ALLOWED_TEMPORAL_RELATIONS:
                cond["temporal_relation"] = cond.get("temporal_relation", comp)
                cond["comparator"] = "="
            else:
                cond["comparator"] = comp
        else:
            cond["comparator"] = comp

    # value_operator is only AND/OR; NOT is handled via status dimensions or condition_logic.
    if "value_operator" in cond:
        op = str(cond.get("value_operator", "")).strip().upper()
        if op in _ALLOWED_VALUE_OPERATORS:
            cond["value_operator"] = op
        else:
            cond.pop("value_operator", None)

    # numeric_threshold range normalization.
    if ctype == "numeric_threshold":
        if "inclusive_min" in cond:
            cond["inclusive_min"] = _coerce_bool(cond.get("inclusive_min"), True)
        if "inclusive_max" in cond:
            cond["inclusive_max"] = _coerce_bool(cond.get("inclusive_max"), True)
        if cond.get("comparator") == "between":
            cond.setdefault("inclusive_min", True)
            cond.setdefault("inclusive_max", True)

    # categorical_state: map old single status to dimensional fields and validate dimensions.
    if ctype == "categorical_state":
        old_status = cond.pop("status", None)
        if old_status is not None:
            status_norm = str(old_status).strip().lower().replace(" ", "_")
            target_field = _STATUS_TO_CATEGORICAL_FIELD.get(status_norm)
            if target_field and target_field not in cond:
                cond[target_field] = status_norm

        if "clinical_status" in cond:
            v = str(cond.get("clinical_status", "")).strip().lower().replace(" ", "_")
            if v in _ALLOWED_CLINICAL_STATUS:
                cond["clinical_status"] = v
            else:
                cond.pop("clinical_status", None)

        if "verification_status" in cond:
            v = str(cond.get("verification_status", "")).strip().lower().replace(" ", "_")
            if v in _ALLOWED_VERIFICATION_STATUS:
                cond["verification_status"] = v
            else:
                cond.pop("verification_status", None)

        if "severity" in cond:
            v = str(cond.get("severity", "")).strip().lower().replace(" ", "_")
            if v in _ALLOWED_SEVERITY:
                cond["severity"] = v
            else:
                cond.pop("severity", None)

    # medication_history: keep and validate medication-specific status.
    if ctype == "medication_history":
        status = str(cond.get("status", "")).strip().lower().replace(" ", "_")
        cond["status"] = status if status in _ALLOWED_MEDICATION_STATUS else (status or "unknown")

    # temporal_condition qualitative relation.
    if ctype == "temporal_condition" and "temporal_relation" in cond:
        tr = str(cond.get("temporal_relation", "")).strip().lower().replace(" ", "_")
        if tr in _ALLOWED_TEMPORAL_RELATIONS:
            cond["temporal_relation"] = tr
        else:
            cond.pop("temporal_relation", None)

    return cond


def _normalize_conditions(conditions: list) -> list[dict]:
    if not isinstance(conditions, list):
        return []
    valid = []
    for cond in conditions:
        if isinstance(cond, dict) and _validate_condition(cond):
            valid.append(_clean_optional_fields(cond))
    return valid


# ──────────────────────────────────────────────────────────────────
# Triple Augmentation Helpers
# ──────────────────────────────────────────────────────────────────

_EMPTY_COND_SOURCE = {"guideline_id": "", "evidence_level": "", "evidence_texts": []}


def _apply_no_conditions(triple: dict) -> None:
    """Empty-condition stamp for triples that didn't pass head/tail filter."""
    triple["conditions"] = []
    triple["condition_logic"] = None
    triple["condition_source"] = dict(_EMPTY_COND_SOURCE)
    triple["recommendation_strength"] = None
    triple["conditions_json"] = "[]"
    triple["has_conditions"] = False
    triple["parse_failed"] = False
    triple["condition_schema_version"] = "condition_schema_v2"


def _apply_parse_failed(triple: dict) -> None:
    """LLM result missing after retry — Stage 4 excludes these from the graph."""
    _apply_no_conditions(triple)
    triple["parse_failed"] = True


_VALID_CONDITION_LOGIC = {"AND", "OR", "NOT"}


def _apply_cr(triple: dict, cr: dict, recs: list[dict]) -> None:
    """Merge an LLM condition result into a triple."""
    valid_conditions = _normalize_conditions(cr.get("conditions", []))
    source = cr.get("condition_source") or {}
    rec0 = recs[0] if recs else {}

    # Drop invalid condition_logic values (model occasionally emits "AND/OR" etc.)
    cl = cr.get("condition_logic", "AND")
    if cl not in _VALID_CONDITION_LOGIC:
        cl = "AND"

    triple["conditions"] = valid_conditions
    triple["condition_logic"] = cl if valid_conditions else None
    triple["condition_source"] = {
        # Fall back to the matched rec's guideline_id so empty-condition triples
        # still carry their source — they were matched bidirectionally to a rec.
        "guideline_id": source.get("guideline_id", "") or rec0.get("guideline_id", ""),
        "evidence_level": source.get("evidence_level", ""),
        "evidence_texts": source.get("evidence_texts", []),
    }
    triple["recommendation_strength"] = rec0.get("strength")
    triple["conditions_json"] = (
        json.dumps(valid_conditions, ensure_ascii=False) if valid_conditions else "[]"
    )
    triple["has_conditions"] = bool(valid_conditions)
    triple["parse_failed"] = False
    triple["condition_schema_version"] = "condition_schema_v2"


# ──────────────────────────────────────────────────────────────────
# Main Stage 3 Runner
# ──────────────────────────────────────────────────────────────────

def run_stage3(
    triples: list[dict],
    recommendations: list[dict],
    entities: list[dict],
    match_results: list[dict],
    batch_size: int = None,
    max_workers: int = None,
) -> list[dict]:
    """Stage 3: Condition Augmentation.

    Args:
        triples: from stage2_umls_layer_triples.json
        recommendations: from stage0_recommendations.json
        entities: from stage1_entity_candidates.json
        match_results: from stage2_umls_matched.json (for CUI-based matching)
        batch_size: triples per LLM call (overrides STAGE3_LLM_CHUNK_SIZE)
        max_workers: parallel LLM workers (defaults to config.LLM_MAX_WORKERS)
    """
    llm_chunk_size = batch_size or config.STAGE3_LLM_CHUNK_SIZE
    max_workers = max_workers or config.LLM_MAX_WORKERS

    logger.info("=" * 60)
    logger.info("STAGE 3: Condition Augmentation")
    logger.info("=" * 60)
    logger.info(f"  Triples: {len(triples)}")
    logger.info(f"  Recommendations: {len(recommendations)}")
    logger.info(f"  Match results: {len(match_results)}")
    logger.info(f"  LLM chunk size: {llm_chunk_size}, workers: {max_workers}")

    rec_index = build_recommendation_index(recommendations, entities, match_results)

    # Step 1: Endpoint-validity check, then group by (endpoints, relation).
    #
    # Endpoint validity: a triple is only sent for condition extraction when
    # both head_name and tail_name are present — otherwise neither rec lookup
    # nor LLM reasoning is meaningful.
    #
    # Group key includes relation so that triples with the same (head, tail)
    # but different relations are evaluated independently. Without relation
    # in the key, one LLM call's result would be broadcast across siblings
    # (e.g. clinical conditions extracted under `may_be_treated_by` would
    # leak onto `SY`/`translation_of`/`PAR`), violating the prompt rule
    # "only attach conditions that actually constrain the triple relation".
    #
    # Rec cache is still keyed by endpoint pair only — finding recs does not
    # depend on relation, so multiple groups sharing the same endpoints
    # share one rec lookup.
    rec_cache: dict = {}              # endpoint_key → list[rec]
    groups: dict = {}                 # group_key (endpoints + relation) → triple indices
    group_to_endpoint: dict = {}      # group_key → endpoint_key (for retrieving recs)
    no_rec_count = 0
    for i, t in enumerate(triples):
        head_name = t.get("head_name", "").lower().strip()
        tail_name = t.get("tail_name", "").lower().strip()

        # Endpoint validity gate — both endpoints must exist.
        if not head_name or not tail_name:
            _apply_no_conditions(t)
            no_rec_count += 1
            continue

        endpoint_key = (
            t.get("head_cui", ""),
            t.get("tail_id", ""),
            head_name,
            tail_name,
        )
        if endpoint_key not in rec_cache:
            rec_cache[endpoint_key] = find_relevant_recommendations(
                t, recommendations, rec_index
            )
        if not rec_cache[endpoint_key]:
            _apply_no_conditions(t)
            no_rec_count += 1
            continue

        group_key = endpoint_key + (t.get("relation", ""),)
        groups.setdefault(group_key, []).append(i)
        group_to_endpoint.setdefault(group_key, endpoint_key)

    group_keys = list(groups.keys())
    group_originals = [groups[k] for k in group_keys]
    rep_triples = [triples[oi[0]] for oi in group_originals]
    rep_recs = [rec_cache[group_to_endpoint[k]] for k in group_keys]

    candidates = sum(len(oi) for oi in group_originals)
    logger.info(f"  Endpoint cache: {len(rec_cache)} unique endpoint pairs")
    logger.info(
        f"  LLM dedup: {candidates} candidate triples → "
        f"{len(rep_triples)} unique (endpoint, relation) groups "
        f"(saved {candidates - len(rep_triples)})"
    )

    # Step 2: Submit one LLM call per group, parallelized.
    rep_results: list[Optional[dict]] = [None] * len(rep_triples)
    if rep_triples:
        chunks = [
            (start, rep_triples[start:start + llm_chunk_size],
             rep_recs[start:start + llm_chunk_size])
            for start in range(0, len(rep_triples), llm_chunk_size)
        ]
        progress_every = max(1, len(chunks) // 50)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(extract_conditions_batch, ct, cr): (cs, len(ct))
                for cs, ct, cr in chunks
            }
            for fut in as_completed(futures):
                chunk_start, chunk_len = futures[fut]
                for r in fut.result():
                    if not isinstance(r, dict):
                        continue
                    local = r.get("triple_index", -1)
                    # Tight bound: must be within THIS chunk's actual length so a
                    # stray index can't overwrite a different chunk's slot.
                    if isinstance(local, int) and 0 <= local < chunk_len:
                        rep_results[chunk_start + local] = r
                completed += 1
                if completed % progress_every == 0 or completed == len(chunks):
                    logger.info(f"  LLM progress: {completed}/{len(chunks)} chunks")

    # Step 3: Apply each group's LLM result to triples sharing the same
    # (endpoints, relation). Recs are looked up via the endpoint key since
    # the rec set depends on endpoints only.
    parse_failed = 0
    with_cond = 0
    total_cond = 0
    for cr, key, orig_indices in zip(rep_results, group_keys, group_originals):
        if cr is None:
            for i in orig_indices:
                _apply_parse_failed(triples[i])
            parse_failed += len(orig_indices)
            continue
        recs = rec_cache[group_to_endpoint[key]]
        for i in orig_indices:
            _apply_cr(triples[i], cr, recs)
            if triples[i]["has_conditions"]:
                with_cond += 1
                total_cond += len(triples[i]["conditions"])

    logger.info("=" * 60)
    logger.info("STAGE 3 COMPLETE")
    logger.info(
        f"  Triples with conditions: {with_cond}/{len(triples)} "
        f"({with_cond / max(len(triples), 1) * 100:.1f}%)"
    )
    logger.info(f"  Total conditions: {total_cond}")
    logger.info(f"  Triples skipped LLM (no head+tail rec match): {no_rec_count}")
    logger.info(f"  LLM calls saved by dedup: {candidates - len(rep_triples)} triples")
    if parse_failed:
        logger.warning(
            f"  Triples with parse failure (excluded from Neo4j): {parse_failed}"
        )
    logger.info("=" * 60)

    return triples
