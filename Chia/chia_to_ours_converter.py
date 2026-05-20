"""
Chia → Ours Translation Pipeline
================================

HuggingFace의 bigbio/chia 데이터셋 (`chia_without_scope_fixed_source` subset)을 로드하여
Ours condition schema에 맞는 quadruple 포맷 `(triple, [condition_list])`로 변환한다.

translation rule:
  - quadruple 프레임: text_type이 triple 결정, entities/relations가 condition_list 채움
  - 4가지 condition type: numeric_threshold, categorical_state, medication_history, temporal_condition
  - 4가지 conversion_status: converted, partial, unconverted, evidence_only
  - 모든 Method 허용값 정규화 적용 (severity, clinical_status, comparator 등)
  - Method 문서 스키마 완전 정합

사전 설치: pip install "datasets==2.21.0" (또는 datasets 3.x + trust_remote_code)
실행:      python chia_to_ours_converter.py
출력:      chia_to_ours_quadruples.json
"""

import re
import json
from collections import defaultdict, Counter
from datasets import load_dataset


# ════════════════════════════════════════════════════════════════
# 1. 변환 규칙용 상수 정의
# ════════════════════════════════════════════════════════════════

# ─── Method 허용값 (validation용) ─────────────────────────────────
METHOD_CLINICAL_STATUS = {
    "active", "inactive", "stable", "unstable", "resolved", "present",
    "absent", "current", "past", "no_history", "unknown",
}
METHOD_VERIFICATION_STATUS = {
    "confirmed", "suspected", "excluded", "contraindicated", "unknown",
}
METHOD_SEVERITY = {"mild", "moderate", "severe"}
METHOD_MED_STATUS = {
    "current", "prior", "past", "failed", "discontinued",
    "contraindicated", "not_current", "unknown",
}
METHOD_COMPARATORS = {"<", "<=", "=", ">=", ">", "between"}
METHOD_TEMPORAL_RELATIONS = {
    "before", "after", "during", "overlaps", "contains",
    "equals", "starts", "finishes", "meets",
}

# ─── Rule ② Negation ───────────────────────────────────────────────
NEGATION_HISTORY_PATTERN = re.compile(r'\b(history|ever|had)\b', re.IGNORECASE)

# ─── Rule ②-1 medication_history status → subtype 1:1 매핑 ─────────
MED_STATUS_TO_SUBTYPE = {
    "current": "current_medication",
    "prior": "prior_medication",
    "past": "prior_medication",
    "failed": "medication_failure",
    "discontinued": "discontinued_medication",
    "contraindicated": "contraindicated_medication",
    # not_current, unknown → subtype 미설정
}

# ─── 개선안 3: sentence_text 기반 medication status 추론 패턴 ─────────
# Translation Rule 문서 및 Method 문서에 명시적으로 존재하는 키워드만 허용.
# hallucinated field 방지: 원문에 명시적 키워드가 존재할 때만 status 설정.
# 패턴은 (정규식, status값) 튜플 리스트. 긴 매치 우선 (리스트 순서대로 적용).
MED_STATUS_TEXT_PATTERNS = [
    # failed (Method 예시: "failed first-line antibiotics" → status="failed")
    (re.compile(r'\b(?:fail(?:ed|ure|ing)?)\b', re.IGNORECASE), "failed"),
    # discontinued (Method Allowed status: "discontinued")
    (re.compile(r'\b(?:discontinu(?:ed|ing|ation)|stopped|stop(?:ping)?|ceased|'
                r'withdrawn|withdrawal)\b', re.IGNORECASE), "discontinued"),
    # contraindicated (Method Allowed status: "contraindicated")
    (re.compile(r'\b(?:contraindicated?|contraindication)\b', re.IGNORECASE), "contraindicated"),
    # not_current (Negation Rule 3: "not taking Drug" → status="not_current")
    (re.compile(r'\b(?:not\s+(?:currently\s+)?(?:taking|receiving|using|on)|'
                r'without\s+(?:current\s+)?(?:use|treatment|therapy)|'
                r'never\s+(?:treated|received|taken|used))\b', re.IGNORECASE), "not_current"),
    # prior (Translation Rule: Drug --HAS_TEMPORAL--> prior/past → status="prior")
    (re.compile(r'\b(?:prior|previous(?:ly)?|former(?:ly)?|past)\s+'
                r'(?:use|treatment|therapy|exposure|administration|medication)\b', re.IGNORECASE), "prior"),
    # current (Translation Rule: Drug --HAS_TEMPORAL--> current → status="current")
    (re.compile(r'\b(?:currently|current(?:ly)?)\s+(?:taking|receiving|using|on)\b', re.IGNORECASE), "current"),
    # current — 단독 동사 (context window ±50자 내에서 Drug 근처에 출현 시)
    # "on" 제거: "on day", "based on", "on the study" 등 오탐 빈발.
    # "taking", "receiving", "using"만 허용 — 임상시험 맥락에서 현재 복용 의미가 명확.
    (re.compile(r'\b(?:taking|receiving|using)\b', re.IGNORECASE), "current"),
]

# ─── Rule ③ Qualifier 정규화 딕셔너리 ──────────────────────────────
SEVERITY_NORMALIZE = {
    "severe": "severe", "severely": "severe",
    "mild": "mild", "mildly": "mild",
    "moderate": "moderate", "moderately": "moderate",
    "serious": "severe", "significant": "severe",
    "clinically significant": "severe", "major": "severe",
}
CLINICAL_STATUS_NORMALIZE = {
    "active": "active", "inactive": "inactive",
    "stable": "stable", "unstable": "unstable",
    "resolved": "resolved", "current": "current", "past": "past",
    "chronic": "active", "acute": "active",
    "progressive": "active", "recurrent": "active", "persistent": "active",
    "controlled": "stable", "uncontrolled": "unstable",
    "symptomatic": "present", "asymptomatic": "absent",
}

# ─── Rule ① Temporal 키워드 ────────────────────────────────────────
TEMPORAL_STATUS_KEYWORDS = {
    "history", "history of", "medical history", "previous", "previously",
    "current", "currently", "concomitant", "baseline", "pre-existing",
    "pre-treatment", "pre-operative", "preoperative", "pre-study",
    "pre-enrollment", "post-operative", "postoperative", "post-bronchodilator",
    "undergoing", "receiving", "newly", "new onset", "newly-diagnosed",
    "ongoing", "recent", "prior", "past", "active", "stable", "resolved",
    "chronic", "acute", "long-term", "still", "have had", "antecedent",
    "comorbid", "longstanding", "sustained",
}
TEMPORAL_STATUS_PREFIXES = ("pre-", "post-")

# Temporal status text → clinical_status 매핑
TEMPORAL_STATUS_TO_CLINICAL_STATUS = {
    "history": "past", "history of": "past", "medical history": "past",
    "previous": "past", "previously": "past", "past": "past", "prior": "past",
    "current": "current", "currently": "current", "concomitant": "current",
    "ongoing": "active", "receiving": "current", "undergoing": "current",
    "newly": "active", "new onset": "active", "newly-diagnosed": "active",
    "recent": "current", "active": "active", "stable": "stable",
    "resolved": "resolved", "chronic": "active", "acute": "active",
    "long-term": "active", "still": "current", "have had": "past",
    "antecedent": "past", "comorbid": "active", "longstanding": "active",
    "sustained": "active", "baseline": "current",
}

# Temporal INTERVAL 패턴 (수 + 시간 단위)
TEMPORAL_INTERVAL_RE = re.compile(
    r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*'
    r'(days?|weeks?|wks?|months?|mo|years?|yrs?|hours?|hrs?|minutes?|mins?|half-li[fv]es)\b',
    re.IGNORECASE,
)
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
UNIT_NORMALIZE = {
    "day": "days", "days": "days",
    "week": "weeks", "weeks": "weeks", "wk": "weeks", "wks": "weeks",
    "month": "months", "months": "months", "mo": "months",
    "year": "years", "years": "years", "yr": "years", "yrs": "years",
    "hour": "hours", "hours": "hours", "hr": "hours", "hrs": "hours",
    "minute": "minutes", "minutes": "minutes", "min": "minutes", "mins": "minutes",
}

# Temporal ANCHOR 패턴 (HAS_INDEX 미사용 시 텍스트에서 추출)
TEMPORAL_ANCHOR_RE = re.compile(
    r'(prior to|before|after|since|during|from|following|throughout|'
    r'for the duration of|up to time of)\s+([\w\s]+?)(?:[,\.\(]|$)',
    re.IGNORECASE,
)
TEMPORAL_RELATION_KEYWORDS = {
    "before": "before", "prior to": "before",
    "after": "after", "following": "after", "since": "after",
    "during": "during", "throughout": "during", "from": "during",
}
# within → comparator <=
WITHIN_RE = re.compile(r'\bwithin\b', re.IGNORECASE)
AT_LEAST_RE = re.compile(r'\bat least\b', re.IGNORECASE)
FOR_RE = re.compile(r'\bfor\b', re.IGNORECASE)

# ─── Rule ⑦ Value.text 3단계 분기 ──────────────────────────────────
# 1단계: 구조화 수치
VALUE_STRUCTURED_RE = re.compile(
    r'^\s*([<>=≤≥!]{1,2})\s*(\d+(?:\.\d+)?)\s*(.*)$'
)
# 1단계: 범위 (숫자-숫자)
VALUE_RANGE_RE = re.compile(
    r'^\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(.*)$'
)
# 1단계: between N and M
VALUE_BETWEEN_RE = re.compile(
    r'between\s+(\d+(?:\.\d+)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?)\s*(.*)',
    re.IGNORECASE,
)
# 2단계: 범주 키워드
VALUE_CATEGORICAL_KW = {
    "positive", "negative", "normal", "abnormal",
    "present", "absent", "+", "-", "yes", "no",
}
# 3단계: 자연어 수치 키워드
VALUE_NATURAL_LANG_RE = re.compile(
    r'\b(greater than or equal to|less than or equal to|at least|at most|'
    r'no more than|no less than|not exceeding|exceeding|greater than|less than|'
    r'more than|fewer than|or older|or younger|or more|or greater|or higher|'
    r'or above|or less|or below|over|under|minimum of|maximum of)\b',
    re.IGNORECASE,
)
# 자연어 키워드 → comparator (긴 매치 우선)
NATURAL_LANG_TO_COMPARATOR = [
    ("greater than or equal to", ">="),
    ("less than or equal to", "<="),
    ("at least", ">="),
    ("at most", "<="),
    ("no more than", "<="),
    ("no less than", ">="),
    ("not exceeding", "<="),
    ("minimum of", ">="),
    ("maximum of", "<="),
    ("greater than", ">"),
    ("less than", "<"),
    ("more than", ">"),
    ("fewer than", "<"),
    ("exceeding", ">"),
    ("or older", ">="),
    ("or younger", "<="),
    ("or more", ">="),
    ("or greater", ">="),
    ("or higher", ">="),
    ("or above", ">="),
    ("or less", "<="),
    ("or below", "<="),
    ("over", ">"),
    ("under", "<"),
]
VALUE_NUMBER_RE = re.compile(r'(\d+(?:\.\d+)?)')

# Comparator 정규화 (≤≥ → <= >=)
COMPARATOR_NORMALIZE = {
    "<": "<", "<=": "<=", "≤": "<=", "=<": "<=",
    ">": ">", ">=": ">=", "≥": ">=", "=>": ">=",
    "=": "=", "==": "=",
    "!=": "!=", "!": "!=",
}

# ─── Rule ⑧ Person 키워드 분류 ─────────────────────────────────────
SEX_KEYWORDS = {
    "male": "male", "males": "male", "men": "male", "man": "male",
    "boy": "male", "boys": "male",
    "female": "female", "females": "female", "women": "female",
    "woman": "female", "girl": "female", "girls": "female",
}
AGE_GROUP_KEYWORDS = {
    "adult", "adults", "pediatric", "children", "child", "elderly",
    "infant", "infants", "adolescent", "adolescents",
    "neonate", "neonates", "geriatric",
}

# ─── Numeric subtype 추론 (Measurement.text 기반) ──────────────────
# Method 허용 subtypes: age, vital_sign, lab_value, score, imaging, duration
# 원문에 명시적으로 존재하는 키워드만 매칭 (hallucinated field 방지)
NUMERIC_SUBTYPE_KEYWORDS = {
    "age": [
        "age", "years old", "years of age", "year old", "year of age",
    ],
    "vital_sign": [
        "blood pressure", "systolic", "diastolic", "heart rate", "pulse",
        "respiratory rate", "temperature", "oxygen saturation", "spo2",
        "bmi", "body mass index", "weight", "height", "body weight",
        "waist circumference", "pulse oximetry",
        # J-2 확장: 폐기능 검사 (FEV1, FVC 등)
        "fev1", "fvc", "fev", "peak flow", "pef", "dlco",
        "lung function", "pulmonary function", "spirometry",
        # J-2 확장: 심전도/혈압 약어
        "qtc", "qtcf", "qtcb", "qt interval",
        "bp",
    ],
    "lab_value": [
        "hemoglobin", "hgb", "hematocrit", "platelet", "wbc",
        "white blood cell", "neutrophil", "lymphocyte",
        "creatinine", "egfr", "gfr", "glomerular filtration",
        "bilirubin", "albumin", "protein", "globulin",
        "alt", "ast", "alkaline phosphatase", "alp", "ggt",
        "inr", "ptt", "aptt", "prothrombin", "fibrinogen",
        "glucose", "hba1c", "a1c", "glycated hemoglobin", "fasting glucose",
        "potassium", "sodium", "calcium", "magnesium", "phosphate",
        "cholesterol", "ldl", "hdl", "triglyceride",
        "tsh", "thyroid",
        "troponin", "bnp", "nt-probnp", "crp", "esr",
        "cd4", "viral load", "hiv rna",
        "urea", "bun", "uric acid", "lactate", "psa",
        "ferritin", "transferrin", "serum iron",
        # J-2 확장: Hb (hemoglobin 약어)
        "hb",
    ],
    "score": [
        "ecog", "karnofsky", "nyha", "child-pugh", "child pugh",
        "meld", "gcs", "glasgow coma", "apache",
        "barthel", "rankin", "modified rankin", "mrs",
        "mmse", "mini-mental", "moca",
        "hamilton", "phq", "gad", "beck depression",
        "vas ", "visual analog", "pain score", "pain scale",
        "performance status", "functional status",
        "apgar", "bishop", "nihss",
        "sofa", "curb-65", "curb65", "wells score",
        "cha2ds2", "chads", "has-bled",
        # J-2 확장: ASA physical status classification
        "asa class", "asa physical status",
    ],
    "imaging": [
        "tumor size", "lesion size", "nodule size", "mass size",
        "tumor diameter", "lesion diameter", "nodule diameter",
        "lvef", "ejection fraction", "lv function",
        "stenosis", "vessel diameter",
        # J-2 확장: 종양 측정
        "longest diameter",
    ],
    "duration": [
        "duration of", "length of stay", "time since",
        "disease duration", "symptom duration",
    ],
}

def infer_numeric_subtype(variable_text):
    """variable text로부터 numeric_threshold.subtype 추론.

    Method 허용 subtypes만 사용. 키워드가 원문에 명시적으로 존재할 때만 매핑.
    불확실하면 None (hallucinated field 방지).
    짧은 키워드(≤4자)는 word boundary로 매칭하여 오탐 방지.
    """
    if not variable_text:
        return None
    t = variable_text.lower()
    # 우선순위: age > score > lab_value > vital_sign > imaging > duration
    for subtype in ("age", "score", "lab_value", "vital_sign", "imaging", "duration"):
        for kw in NUMERIC_SUBTYPE_KEYWORDS[subtype]:
            if len(kw) <= 4:
                # 짧은 약어(alt, ast, gfr, bmi 등)는 word boundary 매칭
                if re.search(r'\b' + re.escape(kw) + r'\b', t):
                    return subtype
            else:
                if kw in t:
                    return subtype
    return None


# ════════════════════════════════════════════════════════════════
# 2. 유틸리티 함수
# ════════════════════════════════════════════════════════════════

def get_entity_text(entity):
    """entity의 text 추출 (list/string 모두 처리)."""
    if not entity:
        return ""
    text = entity.get("text", "")
    if isinstance(text, list):
        return " ".join(text).strip()
    return str(text).strip()


def truncate_evidence(text, max_chars=50):
    """evidence_text를 Method 스펙에 따라 ≤50자로 절단."""
    if not text:
        return ""
    return text[:max_chars]


def normalize_comparator(comp):
    """comparator 기호를 Method 허용값으로 정규화."""
    return COMPARATOR_NORMALIZE.get(comp.strip(), comp.strip())


def comparator_to_inclusive(comp):
    """Rule ⑤: comparator → (inclusive_min, inclusive_max). None은 미적용."""
    if comp == ">=":
        return True, None
    if comp == ">":
        return False, None
    if comp == "<=":
        return None, True
    if comp == "<":
        return None, False
    if comp == "=":
        return True, True
    if comp == "between":
        return True, True
    return None, None


def make_condition_source(document_id, evidence_text, source_entity_ids=None,
                          source_relation_ids=None, subsumes_parent=None,
                          subsumes_relation=None):
    """condition_source 객체 생성 (Rule ⑥, Validation의 source_traceability)."""
    src = {
        "guideline_id": document_id,
        "evidence_level": "chia_annotation",
        "evidence_texts": [evidence_text] if evidence_text else [],
        "source_entity_ids": sorted(set(source_entity_ids or [])),
        "source_relation_ids": sorted(set(source_relation_ids or [])),
    }
    if subsumes_parent:
        src["subsumes_parent"] = subsumes_parent
    if subsumes_relation:
        src["subsumes_relation"] = subsumes_relation
    return src


def parse_value_text(value_text):
    """
    Rule ⑦: Value.text 3단계 분기 파싱.
    Returns (parse_type, fields):
      parse_type ∈ {"structured_numeric", "range_numeric", "between_numeric",
                    "categorical", "natural_numeric", "unparseable"}
    """
    if not value_text:
        return "unparseable", {}

    vt = value_text.strip()
    vt_lower = vt.lower()

    # 1단계-a: operator + number (예: "<8", ">=65")
    m = VALUE_STRUCTURED_RE.match(vt)
    if m:
        comp_raw, num_str, rest = m.groups()
        comp = normalize_comparator(comp_raw)
        if comp in METHOD_COMPARATORS:
            value = float(num_str) if "." in num_str else int(num_str)
            unit = rest.strip() or None
            return "structured_numeric", {
                "comparator": comp,
                "value": value,
                "unit": unit,
            }

    # 1단계-b: 범위 (숫자-숫자) — 단, 단독 숫자(e.g. "5")는 제외
    m = VALUE_RANGE_RE.match(vt)
    if m and "-" in vt or "–" in vt:
        if m:
            v_min_s, v_max_s, rest = m.groups()
            v_min = float(v_min_s) if "." in v_min_s else int(v_min_s)
            v_max = float(v_max_s) if "." in v_max_s else int(v_max_s)
            unit = rest.strip() or None
            return "range_numeric", {
                "comparator": "between",
                "value": f"{v_min_s}-{v_max_s}",
                "value_min": v_min,
                "value_max": v_max,
                "unit": unit,
            }

    # 1단계-c: between N and M
    m = VALUE_BETWEEN_RE.search(vt)
    if m:
        v_min_s, v_max_s, rest = m.groups()
        v_min = float(v_min_s) if "." in v_min_s else int(v_min_s)
        v_max = float(v_max_s) if "." in v_max_s else int(v_max_s)
        unit = rest.strip() or None
        return "between_numeric", {
            "comparator": "between",
            "value": f"{v_min_s}-{v_max_s}",
            "value_min": v_min,
            "value_max": v_max,
            "unit": unit,
        }

    # 2단계: 범주 키워드
    if vt_lower in VALUE_CATEGORICAL_KW:
        return "categorical", {"value": vt}

    # 3단계: 자연어 수치
    if VALUE_NATURAL_LANG_RE.search(vt_lower):
        comparator = None
        for phrase, comp in NATURAL_LANG_TO_COMPARATOR:
            if phrase in vt_lower:
                comparator = comp
                break
        num_match = VALUE_NUMBER_RE.search(vt)
        if comparator and num_match:
            num_str = num_match.group(1)
            value = float(num_str) if "." in num_str else int(num_str)
            return "natural_numeric", {
                "comparator": comparator,
                "value": value,
            }

    return "unparseable", {}


def parse_temporal_text(temp_text):
    """
    Rule ①: Temporal.text 분류.
    Returns (kind, fields):
      kind ∈ {"STATUS", "INTERVAL", "MIXED", "UNCLASSIFIED"}
      fields contains interval, interval_unit, comparator, anchor_from_text,
      clinical_status (for STATUS only).
    """
    if not temp_text:
        return "UNCLASSIFIED", {}

    tt = temp_text.strip()
    tt_lower = tt.lower()
    fields = {}

    # INTERVAL 패턴 매칭
    interval_match = TEMPORAL_INTERVAL_RE.search(tt)
    has_interval = bool(interval_match)
    if has_interval:
        num_str = interval_match.group(1)
        unit_raw = interval_match.group(2).lower()
        if num_str.isdigit():
            interval = int(num_str)
        elif num_str.lower() in NUMBER_WORDS:
            interval = NUMBER_WORDS[num_str.lower()]
        else:
            interval = None
        unit = UNIT_NORMALIZE.get(unit_raw, unit_raw)
        if interval is not None:
            fields["interval"] = interval
            fields["interval_unit"] = unit

        # comparator 추출
        if WITHIN_RE.search(tt_lower):
            fields["comparator"] = "<="
        elif AT_LEAST_RE.search(tt_lower):
            fields["comparator"] = ">="
        elif FOR_RE.search(tt_lower):
            fields["comparator"] = "="
        else:
            fields["comparator"] = "="

        # temporal_relation 추출 (Allen 관계)
        for kw, rel in TEMPORAL_RELATION_KEYWORDS.items():
            if kw in tt_lower:
                fields["temporal_relation"] = rel
                break

        # anchor 텍스트 추출 (HAS_INDEX가 없을 때 보조)
        anchor_match = TEMPORAL_ANCHOR_RE.search(tt)
        if anchor_match:
            fields["anchor_from_text"] = anchor_match.group(2).strip()

    # STATUS 키워드 매칭
    status_kw = None
    for kw in TEMPORAL_STATUS_KEYWORDS:
        if kw in tt_lower:
            status_kw = kw
            break
    if status_kw is None:
        for prefix in TEMPORAL_STATUS_PREFIXES:
            if tt_lower.startswith(prefix):
                status_kw = prefix.rstrip("-")
                break
    has_status = status_kw is not None

    if has_status:
        # 매핑 가능한 clinical_status 결정
        cs = TEMPORAL_STATUS_TO_CLINICAL_STATUS.get(status_kw, "past")
        fields["clinical_status"] = cs

    # 분류 결정 (Rule 4: 혼합 시 INTERVAL 우선)
    if has_interval:
        return ("MIXED" if has_status else "INTERVAL"), fields
    if has_status:
        return "STATUS", fields
    return "UNCLASSIFIED", fields


def classify_person(person_text):
    """Rule ⑧: Person.text 분류. Returns (kind, variable, value)."""
    t = person_text.strip().lower()
    if t in SEX_KEYWORDS:
        return "sex", "sex", SEX_KEYWORDS[t]
    if t in AGE_GROUP_KEYWORDS:
        return "age_group", "age_group", person_text.strip()
    return "demographics", "demographics", person_text.strip()


# ════════════════════════════════════════════════════════════════
# 2-b. 문장 분리 유틸리티
# ════════════════════════════════════════════════════════════════

def split_chia_text(text):
    """Chia 적격 기준 텍스트를 문장(기준 항목) 단위로 분리.

    Chia 데이터셋의 text는 개별 기준 문장이 이어붙여진 형태이므로
    줄바꿈과 공백 누락 경계를 탐지하여 분리한다.

    전략:
      1차: 줄바꿈(\\n)으로 분리
      2차: 줄바꿈 없는 구간에서 공백 누락 경계 탐지
           (소문자+대문자, 숫자+대문자, 닫는괄호+대문자)
      3차: 마침표+공백+대문자 (정상 문장 경계)

    Returns: list of (start_offset, end_offset) tuples
    """
    if not text or not text.strip():
        return [(0, len(text))]

    # 1차: 줄바꿈으로 분리
    segments = []
    if '\n' in text:
        start = 0
        for m in re.finditer(r'\n+', text):
            if m.start() > start:
                segments.append((start, m.start()))
            start = m.end()
        if start < len(text):
            segments.append((start, len(text)))
    else:
        segments = [(0, len(text))]

    # 2차: 각 segment 내에서 공백 누락 경계 추가 탐지
    final = []
    for seg_s, seg_e in segments:
        seg = text[seg_s:seg_e]
        splits = []

        for i in range(1, len(seg)):
            prev_ch = seg[i - 1]
            curr_ch = seg[i]

            if prev_ch.islower() and curr_ch.isupper():
                splits.append(i)
                continue
            if prev_ch.isdigit() and curr_ch.isupper():
                splits.append(i)
                continue
            if prev_ch == ')' and curr_ch.isupper():
                splits.append(i)
                continue
            if i >= 2 and seg[i - 1] == ' ' and seg[i - 2] in '.!':
                if curr_ch.isupper() or curr_ch == '(':
                    splits.append(i)
                    continue

        if not splits:
            final.append((seg_s, seg_e))
        else:
            prev = 0
            for sp in splits:
                if sp > prev:
                    final.append((seg_s + prev, seg_s + sp))
                prev = sp
            if prev < len(seg):
                final.append((seg_s + prev, seg_e))

    # 빈/짧은 문장 병합 (공백 제거 후 5자 미만)
    merged = []
    for s, e in final:
        stripped = text[s:e].strip()
        if len(stripped) < 5 and merged:
            prev_s, prev_e = merged[-1]
            merged[-1] = (prev_s, e)
        elif stripped:
            merged.append((s, e))

    return merged if merged else [(0, len(text))]


def assign_entities_to_sentences(entities, sentence_boundaries):
    """각 entity를 첫 번째 offset 시작 위치 기준으로 문장에 배정.

    Returns: dict  {entity_id: sentence_index}
    """
    sent_map = {}
    for e in entities:
        if not e.get("offsets"):
            continue
        start_pos = e["offsets"][0][0]

        assigned = False
        for idx, (s, end) in enumerate(sentence_boundaries):
            if s <= start_pos < end:
                sent_map[e["id"]] = idx
                assigned = True
                break

        if not assigned:
            min_dist = float('inf')
            best_idx = 0
            for idx, (s, end) in enumerate(sentence_boundaries):
                dist = min(abs(start_pos - s), abs(start_pos - end))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx
            sent_map[e["id"]] = best_idx

    return sent_map


def assign_relations_to_sentences(relations, entity_sent_map):
    """relation을 entity 배정 기준으로 문장에 배정.

    양쪽 entity가 같은 문장이면 해당 문장, 다르면 arg1 기준.
    Returns: dict  {relation_id: sentence_index}
    """
    rel_map = {}
    for r in relations:
        s1 = entity_sent_map.get(r["arg1_id"])
        s2 = entity_sent_map.get(r["arg2_id"])

        if s1 is not None and s2 is not None:
            rel_map[r["id"]] = s1 if s1 == s2 else s1
        elif s1 is not None:
            rel_map[r["id"]] = s1
        elif s2 is not None:
            rel_map[r["id"]] = s2

    return rel_map


# ════════════════════════════════════════════════════════════════
# 3. ChiaToOursConverter — 핵심 변환 클래스
# ════════════════════════════════════════════════════════════════

# 처리할 head entity 우선순위 (Method priority: medication > numeric > temporal > categorical)
HEAD_ENTITY_PRIORITY = ["Drug", "Measurement", "Person", "Temporal",
                        "Condition", "Procedure", "Observation", "Visit", "Device"]


class ChiaToOursConverter:
    def __init__(self):
        self.stats = Counter()

    # ─────────────────────────────────────────────────────────────
    # 메인 진입점
    # ─────────────────────────────────────────────────────────────
    def convert_document(self, row):
        """Chia 단일 record → 문장별 Ours quadruple 리스트.

        하나의 row 내 text를 문장 단위로 분리하여, 각 문장마다 하나의
        quadruple을 생성한다. triple은 text_type에 의해 결정되므로
        같은 row 내 모든 문장이 동일한 triple을 공유한다.

        Returns: list[dict]  — 문장 수만큼의 quadruple 리스트
        """
        document_id = row.get("document_id", row.get("id", "unknown"))
        text_type = (row.get("text_type") or "inclusion").lower()
        if text_type.startswith("inc"):
            text_type = "inclusion"
        elif text_type.startswith("exc"):
            text_type = "exclusion"

        text = row.get("text", "") or ""
        entities = row.get("entities", []) or []
        relations = row.get("relations", []) or []

        # ── Step 1: text_type → triple 프레임 (document-level, 전 문장 공유) ──
        triple = self._build_triple(document_id, text_type, entities)

        # ── Step 2: 문장 분리 + entity/relation 배정 ──
        sentence_boundaries = split_chia_text(text)
        ent_sent_map = assign_entities_to_sentences(entities, sentence_boundaries)
        rel_sent_map = assign_relations_to_sentences(relations, ent_sent_map)

        # ── Step 3: 문장별 quadruple 생성 ──
        quadruples = []
        for sent_idx, (sent_start, sent_end) in enumerate(sentence_boundaries):
            sent_text = text[sent_start:sent_end].strip()

            # 이 문장에 배정된 entity/relation 수집
            sent_entities = [e for e in entities
                            if ent_sent_map.get(e["id"]) == sent_idx]
            sent_relations = [r for r in relations
                             if rel_sent_map.get(r["id"]) == sent_idx]

            # entity dictionary (이 문장 범위)
            entity_dict = {e["id"]: e for e in sent_entities}

            # relation graph (이 문장 범위)
            rel_graph = defaultdict(list)
            for r in sent_relations:
                r_type_norm = r["type"].lower().replace(" ", "_")
                a1, a2 = r.get("arg1_id"), r.get("arg2_id")
                r_id = r.get("id", "")
                if a1 and a1 in entity_dict:
                    rel_graph[a1].append((r_type_norm, a2, r_id, "out"))
                if a2 and a2 in entity_dict:
                    rel_graph[a2].append((r_type_norm, a1, r_id, "in"))

            # condition_logic (이 문장의 relation 기준)
            condition_logic = self._determine_logic(sent_relations)

            # condition_list 생성
            condition_list = self._build_condition_list(
                document_id, entity_dict, rel_graph, sent_relations, sent_text
            )

            # SUBSUMES metadata 적용 (이 문장 범위)
            self._apply_subsumes_metadata(
                condition_list, entity_dict, sent_relations
            )

            # conversion_status 결정
            overall_status = self._compute_overall_status(condition_list)

            self.stats[f"text_type:{text_type}"] += 1
            self.stats[f"overall_status:{overall_status}"] += 1

            quadruples.append({
                "id": f"{row.get('id')}_sent{sent_idx + 1}",
                "document_id": document_id,
                "text_type": text_type,
                "sentence_index": sent_idx + 1,
                "sentence_text": sent_text,
                "triple": triple,
                "condition_list": condition_list,
                "condition_logic": condition_logic,
                "conversion_status": overall_status,
            })

        return quadruples

    # ─────────────────────────────────────────────────────────────
    # Step 2: Quadruple triple 프레임 결정
    # ─────────────────────────────────────────────────────────────
    def _build_triple(self, document_id, text_type, entities):
        """Rule 6 + exclusion 3분기 선택 규칙."""
        if text_type == "inclusion":
            return {
                "head": document_id,
                "relation": "has_target_population",
                "tail": "target_population",
            }

        # text_type == "exclusion"
        entity_types = {e["type"] for e in entities}

        # 우선순위 1: Drug 존재 → has_contraindicated_drug
        if "Drug" in entity_types:
            drug_entities = [e for e in entities if e["type"] == "Drug"]
            tail = get_entity_text(drug_entities[0]) or "excluded_drug"
            return {
                "head": document_id,
                "relation": "has_contraindicated_drug",
                "tail": tail,
            }

        # 우선순위 2: Procedure 존재 → has_contraindicated_effect
        if "Procedure" in entity_types:
            proc_entities = [e for e in entities if e["type"] == "Procedure"]
            tail = get_entity_text(proc_entities[0]) or "excluded_condition"
            return {
                "head": document_id,
                "relation": "has_contraindicated_effect",
                "tail": tail,
            }

        # 우선순위 3 (default): Condition 첫 항목 → has_contraindicated_effect
        cond_entities = [e for e in entities if e["type"] == "Condition"]
        tail = get_entity_text(cond_entities[0]) if cond_entities else "excluded_condition"
        return {
            "head": document_id,
            "relation": "has_contraindicated_effect",
            "tail": tail or "excluded_condition",
        }

    # ─────────────────────────────────────────────────────────────
    # Step 5: AND/OR → condition_logic
    # ─────────────────────────────────────────────────────────────
    def _determine_logic(self, relations):
        """Rule 4: AND/OR 기반 condition_logic 결정."""
        has_or = False
        has_and = False
        for r in relations:
            rt = r["type"].upper()
            if rt == "OR":
                has_or = True
            elif rt == "AND":
                has_and = True
        if has_or and not has_and:
            return "OR"
        return "AND"  # 기본값 (Rule 4)

    # ─────────────────────────────────────────────────────────────
    # Steps 6-10: condition_list 생성
    # ─────────────────────────────────────────────────────────────
    def _build_condition_list(self, document_id, entity_dict, rel_graph, relations,
                              sent_text=""):
        """Domain entity별로 condition object 생성. 우선순위에 따라 처리."""
        condition_list = []
        processed = set()  # 이미 다른 condition에 흡수된 entity_id

        # 우선순위 순서대로 처리
        for entity_type in HEAD_ENTITY_PRIORITY:
            for eid, e in entity_dict.items():
                if eid in processed:
                    continue
                if e["type"] != entity_type:
                    continue

                conds, used = None, set()
                if entity_type == "Drug":
                    cond, used = self._build_medication_history(
                        document_id, eid, e, entity_dict, rel_graph, sent_text
                    )
                    conds = [cond] if cond is not None else []
                elif entity_type == "Measurement":
                    cond, used = self._build_numeric_from_measurement(
                        document_id, eid, e, entity_dict, rel_graph
                    )
                    conds = [cond] if cond is not None else []
                elif entity_type == "Person":
                    cond, used = self._build_from_person(
                        document_id, eid, e, entity_dict, rel_graph
                    )
                    conds = [cond] if cond is not None else []
                elif entity_type == "Temporal":
                    # Temporal은 INTERVAL 패턴이고 head로 사용 가능한 경우만
                    cond, used = self._build_temporal_condition(
                        document_id, eid, e, entity_dict, rel_graph
                    )
                    conds = [cond] if cond is not None else []
                else:  # Condition, Procedure, Observation, Visit, Device
                    # 개선안 1: 리스트 반환 (INTERVAL temporal 분리 생성 포함)
                    conds, used = self._build_categorical_state(
                        document_id, eid, e, entity_dict, rel_graph, entity_type
                    )

                if conds:
                    for c in conds:
                        if c is not None:
                            condition_list.append(c)
                    processed.update(used)

        return condition_list

    # ─────────────────────────────────────────────────────────────
    # 헬퍼: 특정 entity에서 나가는 relation으로 partner 찾기
    # ─────────────────────────────────────────────────────────────
    def _get_outgoing(self, eid, rel_graph, rel_type, partner_type=None, entity_dict=None):
        """eid에서 나가는 rel_type relation의 partner들 반환.
        Returns list of (partner_eid, rel_id, partner_entity)."""
        results = []
        rel_type_norm = rel_type.lower()
        for r_type, partner_id, r_id, direction in rel_graph.get(eid, []):
            if direction != "out":
                continue
            if r_type != rel_type_norm:
                continue
            if partner_type and entity_dict:
                partner = entity_dict.get(partner_id)
                if not partner or partner["type"] != partner_type:
                    continue
                results.append((partner_id, r_id, partner))
            else:
                partner = entity_dict.get(partner_id) if entity_dict else None
                results.append((partner_id, r_id, partner))
        return results

    # ─────────────────────────────────────────────────────────────
    # medication_history 생성 (Drug + 관련 entities)
    # ─────────────────────────────────────────────────────────────
    def _build_medication_history(self, document_id, drug_eid, drug_entity,
                                  entity_dict, rel_graph, sent_text=""):
        drug_text = get_entity_text(drug_entity)
        if not drug_text:
            return None, set()

        used = {drug_eid}
        used_rels = set()
        evidence_parts = [drug_text]
        extra_evidence = []

        cond = {
            "type": "medication_history",
            "drug": drug_text,
            "status": "unknown",  # Rule ⑨ standalone 기본값
            "qualifies": "relation",  # Rule 8
        }
        # subtype은 status가 결정되면 추가됨 (Rule ②-1)

        # ── relation 기반 처리 (기존 로직) ──

        # HAS_TEMPORAL → status
        for partner_id, rel_id, partner in self._get_outgoing(
            drug_eid, rel_graph, "has_temporal", "Temporal", entity_dict
        ):
            t_text = get_entity_text(partner)
            kind, fields = parse_temporal_text(t_text)
            used.add(partner_id)
            used_rels.add(rel_id)
            evidence_parts.append(t_text)

            if kind in ("STATUS", "MIXED") and "clinical_status" in fields:
                # clinical_status → medication status로 매핑
                cs = fields["clinical_status"]
                if cs == "past":
                    cond["status"] = "prior"
                elif cs == "current":
                    cond["status"] = "current"
                elif cs == "active":
                    cond["status"] = "current"
                else:
                    cond["status"] = "current"
            else:
                # INTERVAL only or UNCLASSIFIED — evidence_text에 보존
                extra_evidence.append(t_text)

        # HAS_NEGATION → status="not_current"
        for partner_id, rel_id, partner in self._get_outgoing(
            drug_eid, rel_graph, "has_negation", "Negation", entity_dict
        ):
            n_text = get_entity_text(partner)
            used.add(partner_id)
            used_rels.add(rel_id)
            evidence_parts.append(n_text)
            cond["status"] = "not_current"

        # HAS_MULTIPLIER → dose/frequency
        for partner_id, rel_id, partner in self._get_outgoing(
            drug_eid, rel_graph, "has_multiplier", "Multiplier", entity_dict
        ):
            m_text = get_entity_text(partner)
            used.add(partner_id)
            used_rels.add(rel_id)
            evidence_parts.append(m_text)

            # 용량 패턴 (숫자 + 단위)
            dose_match = re.search(r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|μg|ml|l|iu)\b',
                                   m_text, re.IGNORECASE)
            freq_match = re.search(
                r'(once|twice|three times|qd|bid|tid|qid|q\d+h|daily|weekly|'
                r'monthly|per day|per week|per month|\d+\s*times)',
                m_text, re.IGNORECASE,
            )
            if dose_match:
                cond["dose"] = dose_match.group(1)
                cond["unit"] = dose_match.group(2).lower()
            if freq_match:
                cond["frequency"] = freq_match.group(0).strip()
            if not dose_match and not freq_match:
                # 파싱 실패 → evidence_text에 보존
                extra_evidence.append(m_text)

        # HAS_VALUE → dose (수치) 또는 evidence_text (범주)
        for partner_id, rel_id, partner in self._get_outgoing(
            drug_eid, rel_graph, "has_value", "Value", entity_dict
        ):
            v_text = get_entity_text(partner)
            used.add(partner_id)
            used_rels.add(rel_id)
            evidence_parts.append(v_text)
            parse_kind, fields = parse_value_text(v_text)
            if parse_kind in ("structured_numeric", "range_numeric",
                              "between_numeric", "natural_numeric") and "value" in fields:
                if "dose" not in cond:
                    cond["dose"] = str(fields["value"])
                    if fields.get("unit"):
                        cond["unit"] = fields["unit"]
            else:
                extra_evidence.append(v_text)

        # HAS_QUALIFIER → evidence_text 보존 (medication에는 severity/clinical_status 없음)
        for partner_id, rel_id, partner in self._get_outgoing(
            drug_eid, rel_graph, "has_qualifier", "Qualifier", entity_dict
        ):
            q_text = get_entity_text(partner)
            used.add(partner_id)
            used_rels.add(rel_id)
            extra_evidence.append(q_text)
            evidence_parts.append(q_text)

        # HAS_MOOD → evidence_text 보존 (Rule ④)
        mood_present = False
        for partner_id, rel_id, partner in self._get_outgoing(
            drug_eid, rel_graph, "has_mood", "Mood", entity_dict
        ):
            mood_present = True
            m_text = get_entity_text(partner)
            used.add(partner_id)
            used_rels.add(rel_id)
            extra_evidence.append(m_text)
            evidence_parts.append(m_text)

        # ── 개선안 3: sentence_text 기반 status 추론 (relation 미존재 시) ──
        # status가 여전히 "unknown"이면, sentence_text에서 Drug 텍스트 주변의
        # 명시적 키워드를 매칭하여 status를 설정한다.
        # hallucinated field 방지: 원문에 키워드가 존재할 때만 설정.
        if cond["status"] == "unknown" and sent_text and drug_text:
            # Drug 텍스트 주변 context window 추출 (±50자)
            drug_lower = drug_text.lower()
            sent_lower = sent_text.lower()
            drug_pos = sent_lower.find(drug_lower)
            if drug_pos >= 0:
                ctx_start = max(0, drug_pos - 50)
                ctx_end = min(len(sent_text), drug_pos + len(drug_text) + 50)
                context = sent_lower[ctx_start:ctx_end]
            else:
                # Drug 텍스트를 못 찾으면 전체 문장 사용
                context = sent_lower

            for pattern, status_val in MED_STATUS_TEXT_PATTERNS:
                if pattern.search(context):
                    cond["status"] = status_val
                    break

        # Rule ②-1: status → subtype 1:1 매핑
        subtype = MED_STATUS_TO_SUBTYPE.get(cond["status"])
        if subtype:
            cond["subtype"] = subtype

        # evidence_text 및 condition_source
        evidence_text = truncate_evidence(" ".join(evidence_parts))
        cond["evidence_text"] = evidence_text
        cond["condition_source"] = make_condition_source(
            document_id, evidence_text,
            source_entity_ids=list(used),
            source_relation_ids=list(used_rels),
        )

        # conversion_status 결정
        if cond["status"] == "unknown" and not subtype and not cond.get("dose"):
            cond["conversion_status"] = "partial"  # status는 unknown이지만 drug만 있음
        elif extra_evidence:
            cond["conversion_status"] = "partial"  # 일부는 evidence_text로 fallback
        else:
            cond["conversion_status"] = "converted"

        return cond, used

    # ─────────────────────────────────────────────────────────────
    # numeric_threshold (Measurement + Value)
    # ─────────────────────────────────────────────────────────────
    def _build_numeric_from_measurement(self, document_id, meas_eid, meas_entity,
                                        entity_dict, rel_graph):
        meas_text = get_entity_text(meas_entity)
        if not meas_text:
            return None, set()

        used = {meas_eid}
        used_rels = set()
        evidence_parts = [meas_text]
        extra_evidence = []

        # HAS_VALUE → comparator/value/unit
        value_partners = self._get_outgoing(
            meas_eid, rel_graph, "has_value", "Value", entity_dict
        )

        if not value_partners:
            # Value 없음 → numeric_threshold 생성 불가 (Rule ⑨ Measurement standalone)
            # evidence_text만 보존
            evidence_text = truncate_evidence(meas_text)
            cond = {
                "type": "numeric_threshold",
                "variable": meas_text,
                "comparator": "=",  # 임시 (필수 필드)
                "evidence_text": evidence_text,
                "qualifies": "relation",
                "conversion_status": "unconverted",
                "condition_source": make_condition_source(
                    document_id, evidence_text,
                    source_entity_ids=[meas_eid],
                ),
            }
            # Validation: numeric_threshold 필수 필드 누락 시 categorical_state로 fallback
            cond_alt = {
                "type": "categorical_state",
                "subtype": "clinical_state",
                "variable": meas_text,
                "value": meas_text,
                "evidence_text": evidence_text,
                "qualifies": "relation",
                "conversion_status": "evidence_only",
                "condition_source": make_condition_source(
                    document_id, evidence_text,
                    source_entity_ids=[meas_eid],
                ),
            }
            return cond_alt, used

        # 첫 Value를 메인으로 사용
        first_partner_id, first_rel_id, first_partner = value_partners[0]
        value_text = get_entity_text(first_partner)
        used.add(first_partner_id)
        used_rels.add(first_rel_id)
        evidence_parts.append(value_text)

        parse_kind, fields = parse_value_text(value_text)

        # 추가 Value들은 evidence_text에 보존
        for pid, rid, p in value_partners[1:]:
            used.add(pid)
            used_rels.add(rid)
            vt = get_entity_text(p)
            extra_evidence.append(vt)
            evidence_parts.append(vt)

        if parse_kind in ("structured_numeric", "range_numeric",
                          "between_numeric", "natural_numeric"):
            cond = {
                "type": "numeric_threshold",
                "variable": meas_text,
                "comparator": fields["comparator"],
                "qualifies": "relation",
            }
            if "value" in fields:
                cond["value"] = fields["value"]
            if "value_min" in fields:
                cond["value_min"] = fields["value_min"]
            if "value_max" in fields:
                cond["value_max"] = fields["value_max"]
            if fields.get("unit"):
                cond["unit"] = fields["unit"]

            # Rule ⑤: comparator → inclusive_min/max
            inc_min, inc_max = comparator_to_inclusive(fields["comparator"])
            if inc_min is not None:
                cond["inclusive_min"] = inc_min
            if inc_max is not None:
                cond["inclusive_max"] = inc_max

            # subtype 추론
            subtype = infer_numeric_subtype(meas_text)
            if subtype:
                cond["subtype"] = subtype

            # HAS_TEMPORAL, HAS_QUALIFIER 등도 흡수
            self._absorb_modifiers_to_evidence(
                meas_eid, rel_graph, entity_dict, used, used_rels,
                evidence_parts, extra_evidence,
                skip_rel_types={"has_value"},
            )

            evidence_text = truncate_evidence(" ".join(evidence_parts))
            cond["evidence_text"] = evidence_text
            cond["condition_source"] = make_condition_source(
                document_id, evidence_text,
                source_entity_ids=list(used),
                source_relation_ids=list(used_rels),
            )

            status = "partial" if (parse_kind == "natural_numeric" or extra_evidence) else "converted"
            cond["conversion_status"] = status
            return cond, used

        elif parse_kind == "categorical":
            # categorical_state로 변환
            cond = {
                "type": "categorical_state",
                "subtype": "clinical_state",
                "variable": meas_text,
                "value": fields["value"],
                "qualifies": "relation",
            }
            self._absorb_modifiers_to_evidence(
                meas_eid, rel_graph, entity_dict, used, used_rels,
                evidence_parts, extra_evidence,
                skip_rel_types={"has_value"},
            )
            evidence_text = truncate_evidence(" ".join(evidence_parts))
            cond["evidence_text"] = evidence_text
            cond["condition_source"] = make_condition_source(
                document_id, evidence_text,
                source_entity_ids=list(used),
                source_relation_ids=list(used_rels),
            )
            cond["conversion_status"] = "converted"
            return cond, used

        else:  # unparseable
            evidence_text = truncate_evidence(" ".join(evidence_parts))
            cond = {
                "type": "categorical_state",
                "subtype": "clinical_state",
                "variable": meas_text,
                "value": value_text,
                "qualifies": "relation",
                "evidence_text": evidence_text,
                "conversion_status": "unconverted",
                "condition_source": make_condition_source(
                    document_id, evidence_text,
                    source_entity_ids=list(used),
                    source_relation_ids=list(used_rels),
                ),
            }
            return cond, used

    # ─────────────────────────────────────────────────────────────
    # Person → numeric_threshold(age) or categorical_state(population)
    # ─────────────────────────────────────────────────────────────
    def _build_from_person(self, document_id, person_eid, person_entity,
                           entity_dict, rel_graph):
        person_text = get_entity_text(person_entity)
        if not person_text:
            return None, set()

        used = {person_eid}
        used_rels = set()
        evidence_parts = [person_text]
        extra_evidence = []

        # HAS_VALUE 확인
        value_partners = self._get_outgoing(
            person_eid, rel_graph, "has_value", "Value", entity_dict
        )

        if value_partners:
            first_pid, first_rid, first_partner = value_partners[0]
            value_text = get_entity_text(first_partner)
            used.add(first_pid)
            used_rels.add(first_rid)
            evidence_parts.append(value_text)

            parse_kind, fields = parse_value_text(value_text)

            if parse_kind in ("structured_numeric", "range_numeric",
                              "between_numeric", "natural_numeric"):
                # 숫자 포함 → age numeric_threshold
                cond = {
                    "type": "numeric_threshold",
                    "subtype": "age",
                    "variable": "age",
                    "comparator": fields["comparator"],
                    "qualifies": "relation",
                }
                if "value" in fields:
                    cond["value"] = fields["value"]
                if "value_min" in fields:
                    cond["value_min"] = fields["value_min"]
                if "value_max" in fields:
                    cond["value_max"] = fields["value_max"]
                cond["unit"] = fields.get("unit") or "years"

                inc_min, inc_max = comparator_to_inclusive(fields["comparator"])
                if inc_min is not None:
                    cond["inclusive_min"] = inc_min
                if inc_max is not None:
                    cond["inclusive_max"] = inc_max

                self._absorb_modifiers_to_evidence(
                    person_eid, rel_graph, entity_dict, used, used_rels,
                    evidence_parts, extra_evidence,
                    skip_rel_types={"has_value"},
                )
                evidence_text = truncate_evidence(" ".join(evidence_parts))
                cond["evidence_text"] = evidence_text
                cond["condition_source"] = make_condition_source(
                    document_id, evidence_text,
                    source_entity_ids=list(used),
                    source_relation_ids=list(used_rels),
                )
                cond["conversion_status"] = ("partial" if parse_kind == "natural_numeric"
                                              or extra_evidence else "converted")
                return cond, used

            # 숫자 미포함 → categorical_state population
            cond = {
                "type": "categorical_state",
                "subtype": "population",
                "variable": "demographics",
                "value": value_text,
                "qualifies": "relation",
            }
            self._absorb_modifiers_to_evidence(
                person_eid, rel_graph, entity_dict, used, used_rels,
                evidence_parts, extra_evidence,
                skip_rel_types={"has_value"},
            )
            evidence_text = truncate_evidence(" ".join(evidence_parts))
            cond["evidence_text"] = evidence_text
            cond["condition_source"] = make_condition_source(
                document_id, evidence_text,
                source_entity_ids=list(used),
                source_relation_ids=list(used_rels),
            )
            cond["conversion_status"] = "converted"
            return cond, used

        # HAS_VALUE 없음 → Person.text 기반 키워드 분류 (Rule ⑧)
        kind, variable, value = classify_person(person_text)
        cond = {
            "type": "categorical_state",
            "subtype": "population",
            "variable": variable,
            "value": value,
            "qualifies": "relation",
        }
        self._absorb_modifiers_to_evidence(
            person_eid, rel_graph, entity_dict, used, used_rels,
            evidence_parts, extra_evidence,
        )
        evidence_text = truncate_evidence(" ".join(evidence_parts))
        cond["evidence_text"] = evidence_text
        cond["condition_source"] = make_condition_source(
            document_id, evidence_text,
            source_entity_ids=list(used),
            source_relation_ids=list(used_rels),
        )
        cond["conversion_status"] = "converted" if kind in ("sex", "age_group") else "partial"
        return cond, used

    # ─────────────────────────────────────────────────────────────
    # Temporal → temporal_condition (INTERVAL 패턴 인 경우만)
    # ─────────────────────────────────────────────────────────────
    def _build_temporal_condition(self, document_id, temp_eid, temp_entity,
                                  entity_dict, rel_graph):
        temp_text = get_entity_text(temp_entity)
        if not temp_text:
            return None, set()

        kind, fields = parse_temporal_text(temp_text)

        # STATUS 또는 UNCLASSIFIED인 경우, 자체 temporal_condition 생성하지 않고
        # parent에 흡수되도록 처리 (skip, parent가 처리)
        if kind == "STATUS" or kind == "UNCLASSIFIED":
            return None, set()

        # 개선안 1 연동: 이 Temporal entity에 incoming has_temporal이 있고,
        # parent가 categorical entity type(Condition/Procedure/Observation/Visit/Device)이면
        # _build_categorical_state()가 INTERVAL 분리 생성을 처리하므로 여기서는 skip.
        # (중복 temporal_condition 생성 방지)
        _CATEGORICAL_TYPES = {"Condition", "Procedure", "Observation", "Visit", "Device"}
        for r_type, partner_id, r_id, direction in rel_graph.get(temp_eid, []):
            if direction == "in" and r_type == "has_temporal":
                parent = entity_dict.get(partner_id)
                if parent and parent.get("type") in _CATEGORICAL_TYPES:
                    return None, set()  # parent의 _build_categorical_state가 처리

        # INTERVAL 또는 MIXED — temporal_condition 생성
        used = {temp_eid}
        used_rels = set()
        evidence_parts = [temp_text]

        # event 결정: HAS_TEMPORAL의 arg1(parent)에서 가져옴
        event = None
        for r_type, partner_id, r_id, direction in rel_graph.get(temp_eid, []):
            if direction == "in" and r_type == "has_temporal":
                partner = entity_dict.get(partner_id)
                if partner:
                    event = get_entity_text(partner)
                    used_rels.add(r_id)
                    break

        if not event:
            event = temp_text  # fallback

        # anchor 결정: HAS_INDEX → Reference_point
        anchor = ""  # 기본값 (Temporal Rule 6)
        anchor_partners = self._get_outgoing(
            temp_eid, rel_graph, "has_index", "Reference_point", entity_dict
        )
        if anchor_partners:
            anchor_pid, anchor_rid, anchor_p = anchor_partners[0]
            anchor = get_entity_text(anchor_p)
            used.add(anchor_pid)
            used_rels.add(anchor_rid)
            evidence_parts.append(anchor)
        elif "anchor_from_text" in fields:
            anchor = fields["anchor_from_text"]

        cond = {
            "type": "temporal_condition",
            "event": event,
            "anchor": anchor,
            "comparator": fields.get("comparator", "="),
            "qualifies": "relation",
        }
        if "interval" in fields:
            cond["interval"] = fields["interval"]
        if "interval_unit" in fields:
            cond["interval_unit"] = fields["interval_unit"]
        if "temporal_relation" in fields:
            tr = fields["temporal_relation"]
            if tr in METHOD_TEMPORAL_RELATIONS:
                cond["temporal_relation"] = tr

        evidence_text = truncate_evidence(" ".join(evidence_parts))
        cond["evidence_text"] = evidence_text
        cond["condition_source"] = make_condition_source(
            document_id, evidence_text,
            source_entity_ids=list(used),
            source_relation_ids=list(used_rels),
        )

        if cond["comparator"] in METHOD_COMPARATORS and "interval" in cond:
            cond["conversion_status"] = "converted"
        else:
            cond["conversion_status"] = "partial"

        return cond, used

    # ─────────────────────────────────────────────────────────────
    # categorical_state (Condition, Procedure, Observation, Visit, Device)
    # 개선안 1: INTERVAL/MIXED temporal → 별도 temporal_condition 분리 생성
    # Returns: (cond_list, used) — cond_list는 list[dict]
    # ─────────────────────────────────────────────────────────────
    def _build_categorical_state(self, document_id, ent_eid, entity,
                                 entity_dict, rel_graph, entity_type):
        ent_text = get_entity_text(entity)
        if not ent_text:
            return [], set()

        used = {ent_eid}
        used_rels = set()
        evidence_parts = [ent_text]

        # entity type별 기본 subtype (Rule ⑨)
        subtype_map = {
            "Condition": "clinical_state",
            "Procedure": "procedure_history",
            "Observation": "patient_context",
            "Visit": "care_setting",
            "Device": "intervention_status",
        }
        # entity type별 기본 clinical_status (Rule ⑨)
        default_cs = {
            "Condition": "present",
            "Procedure": "unknown",
            "Observation": None,  # 미설정
            "Visit": None,
            "Device": None,
        }

        cond = {
            "type": "categorical_state",
            "subtype": subtype_map.get(entity_type, "clinical_state"),
            "variable": ent_text,
            "value": ent_text,
            "qualifies": "relation",
        }
        if default_cs.get(entity_type):
            cond["clinical_status"] = default_cs[entity_type]

        had_modifier = False  # 어떤 modifier가 흡수되었는지 추적
        partial_evidence = []  # evidence_text로만 보존된 내용
        spawned_temporal_conds = []  # 분리 생성된 temporal_condition들

        # HAS_QUALIFIER → severity / clinical_status (Rule ③)
        for partner_id, rel_id, partner in self._get_outgoing(
            ent_eid, rel_graph, "has_qualifier", "Qualifier", entity_dict
        ):
            q_text = get_entity_text(partner)
            q_lower = q_text.strip().lower()
            used.add(partner_id)
            used_rels.add(rel_id)
            evidence_parts.append(q_text)

            if q_lower in SEVERITY_NORMALIZE:
                if "severity" not in cond:
                    cond["severity"] = SEVERITY_NORMALIZE[q_lower]
                    had_modifier = True
                else:
                    partial_evidence.append(q_text)  # 충돌 시 evidence_text 보존
            elif q_lower in CLINICAL_STATUS_NORMALIZE:
                new_cs = CLINICAL_STATUS_NORMALIZE[q_lower]
                # 기본값이 아닌 경우만 덮어쓰기, 또는 첫 매칭만 유지
                if cond.get("clinical_status") in (None, default_cs.get(entity_type)):
                    cond["clinical_status"] = new_cs
                    had_modifier = True
                else:
                    partial_evidence.append(q_text)
            else:
                partial_evidence.append(q_text)  # evidence_only

        # HAS_NEGATION → clinical_status="absent"/"no_history" (Rule ②)
        for partner_id, rel_id, partner in self._get_outgoing(
            ent_eid, rel_graph, "has_negation", "Negation", entity_dict
        ):
            n_text = get_entity_text(partner)
            used.add(partner_id)
            used_rels.add(rel_id)
            evidence_parts.append(n_text)

            # "no history of" 패턴 체크
            if NEGATION_HISTORY_PATTERN.search(n_text) or "history" in ent_text.lower():
                cond["clinical_status"] = "no_history"
            else:
                cond["clinical_status"] = "absent"
            had_modifier = True

        # HAS_TEMPORAL → STATUS이면 clinical_status 흡수,
        # INTERVAL/MIXED이면 별도 temporal_condition 분리 생성 (개선안 1)
        for partner_id, rel_id, partner in self._get_outgoing(
            ent_eid, rel_graph, "has_temporal", "Temporal", entity_dict
        ):
            t_text = get_entity_text(partner)
            kind, fields = parse_temporal_text(t_text)
            used.add(partner_id)
            used_rels.add(rel_id)
            evidence_parts.append(t_text)

            if kind == "STATUS" and "clinical_status" in fields:
                # STATUS만이면 parent의 clinical_status로 흡수
                if cond.get("clinical_status") in (None, default_cs.get(entity_type)):
                    cond["clinical_status"] = fields["clinical_status"]
                else:
                    partial_evidence.append(t_text)
                had_modifier = True

            elif kind in ("INTERVAL", "MIXED"):
                # ── 개선안 1: INTERVAL/MIXED → 별도 temporal_condition 생성 ──
                # Translation Rule Step 8, Temporal Rule 2: INTERVAL 패턴 →
                # temporal_condition(event=parent_entity.text) 으로 변환
                # Translation Rule: "Procedure/Condition --HAS_TEMPORAL--> within X"
                # → temporal_condition(event=parent_entity.text, comparator="<=", interval=X)

                # MIXED인 경우: STATUS 부분만 parent에 흡수
                if kind == "MIXED" and "clinical_status" in fields:
                    if cond.get("clinical_status") in (None, default_cs.get(entity_type)):
                        cond["clinical_status"] = fields["clinical_status"]
                    had_modifier = True

                # temporal_condition 생성 (INTERVAL 부분)
                if "interval" in fields:
                    tc_used = {partner_id}
                    tc_used_rels = {rel_id}
                    tc_evidence_parts = [t_text]

                    # event = parent entity text (Translation Rule 패턴)
                    tc_event = ent_text

                    # anchor 결정: Temporal entity의 HAS_INDEX → Reference_point
                    tc_anchor = ""
                    anchor_partners = self._get_outgoing(
                        partner_id, rel_graph, "has_index",
                        "Reference_point", entity_dict
                    )
                    if anchor_partners:
                        anchor_pid, anchor_rid, anchor_p = anchor_partners[0]
                        tc_anchor = get_entity_text(anchor_p)
                        used.add(anchor_pid)
                        tc_used.add(anchor_pid)
                        tc_used_rels.add(anchor_rid)
                        tc_evidence_parts.append(tc_anchor)
                    elif "anchor_from_text" in fields:
                        tc_anchor = fields["anchor_from_text"]

                    tc_cond = {
                        "type": "temporal_condition",
                        "event": tc_event,
                        "anchor": tc_anchor,
                        "comparator": fields.get("comparator", "="),
                        "qualifies": "relation",
                    }
                    if "interval" in fields:
                        tc_cond["interval"] = fields["interval"]
                    if "interval_unit" in fields:
                        tc_cond["interval_unit"] = fields["interval_unit"]
                    if "temporal_relation" in fields:
                        tr = fields["temporal_relation"]
                        if tr in METHOD_TEMPORAL_RELATIONS:
                            tc_cond["temporal_relation"] = tr

                    tc_evidence_text = truncate_evidence(
                        " ".join(tc_evidence_parts)
                    )
                    tc_cond["evidence_text"] = tc_evidence_text
                    tc_cond["condition_source"] = make_condition_source(
                        document_id, tc_evidence_text,
                        source_entity_ids=list(tc_used),
                        source_relation_ids=list(tc_used_rels),
                    )

                    if (tc_cond["comparator"] in METHOD_COMPARATORS
                            and "interval" in tc_cond):
                        tc_cond["conversion_status"] = "converted"
                    else:
                        tc_cond["conversion_status"] = "partial"

                    spawned_temporal_conds.append(tc_cond)
                    # INTERVAL을 별도 condition으로 분리했으므로
                    # partial_evidence에 추가하지 않음
                else:
                    # interval 파싱 실패 → evidence_text fallback
                    partial_evidence.append(t_text)

            else:
                # UNCLASSIFIED
                partial_evidence.append(t_text)

        # HAS_VALUE → value 또는 evidence_text
        for partner_id, rel_id, partner in self._get_outgoing(
            ent_eid, rel_graph, "has_value", "Value", entity_dict
        ):
            v_text = get_entity_text(partner)
            used.add(partner_id)
            used_rels.add(rel_id)
            evidence_parts.append(v_text)
            parse_kind, fields = parse_value_text(v_text)
            if parse_kind == "categorical":
                # value를 덮어씀
                cond["value"] = f"{ent_text} {v_text}".strip()
                had_modifier = True
            else:
                partial_evidence.append(v_text)

        # HAS_CONTEXT → family history 등 (Rule)
        for partner_id, rel_id, partner in self._get_outgoing(
            ent_eid, rel_graph, "has_context", "Observation", entity_dict
        ):
            o_text = get_entity_text(partner)
            used.add(partner_id)
            used_rels.add(rel_id)
            evidence_parts.append(o_text)
            if "family history" in o_text.lower():
                cond["clinical_status"] = "past"
                cond["value"] = f"family history of {ent_text}"
                cond["subtype"] = "patient_context"
                had_modifier = True
            else:
                cond["value"] = f"{ent_text} ({o_text})"
                had_modifier = True

        # HAS_MULTIPLIER → evidence_text (categorical에는 dose 없음)
        for partner_id, rel_id, partner in self._get_outgoing(
            ent_eid, rel_graph, "has_multiplier", "Multiplier", entity_dict
        ):
            m_text = get_entity_text(partner)
            used.add(partner_id)
            used_rels.add(rel_id)
            evidence_parts.append(m_text)
            partial_evidence.append(m_text)

        # HAS_MOOD → evidence_text 보존만 (Rule ④)
        for partner_id, rel_id, partner in self._get_outgoing(
            ent_eid, rel_graph, "has_mood", "Mood", entity_dict
        ):
            m_text = get_entity_text(partner)
            used.add(partner_id)
            used_rels.add(rel_id)
            evidence_parts.append(m_text)
            partial_evidence.append(m_text)

        evidence_text = truncate_evidence(" ".join(evidence_parts))
        cond["evidence_text"] = evidence_text
        cond["condition_source"] = make_condition_source(
            document_id, evidence_text,
            source_entity_ids=list(used),
            source_relation_ids=list(used_rels),
        )

        # conversion_status 결정
        if partial_evidence:
            cond["conversion_status"] = "partial"
        else:
            cond["conversion_status"] = "converted"

        # 반환: [main_cond, *spawned_temporal_conds]
        result = [cond] + spawned_temporal_conds
        return result, used

    # ─────────────────────────────────────────────────────────────
    # Modifier 흡수 (evidence_text 보존만, severity/status는 미설정)
    # ─────────────────────────────────────────────────────────────
    def _absorb_modifiers_to_evidence(self, eid, rel_graph, entity_dict,
                                       used, used_rels, evidence_parts,
                                       extra_evidence, skip_rel_types=None):
        """numeric_threshold/medication_history 등에서 흡수할 수 없는 modifier들을
        evidence_text에 보존."""
        skip = skip_rel_types or set()
        for r_type, partner_id, r_id, direction in rel_graph.get(eid, []):
            if direction != "out":
                continue
            if r_type in skip:
                continue
            if r_type not in {"has_qualifier", "has_temporal", "has_negation",
                              "has_mood", "has_multiplier", "has_context"}:
                continue
            partner = entity_dict.get(partner_id)
            if not partner:
                continue
            ptext = get_entity_text(partner)
            used.add(partner_id)
            used_rels.add(r_id)
            evidence_parts.append(ptext)
            extra_evidence.append(ptext)

    # ─────────────────────────────────────────────────────────────
    # SUBSUMES metadata 적용 (Rule ⑥)
    # ─────────────────────────────────────────────────────────────
    def _apply_subsumes_metadata(self, condition_list, entity_dict, relations):
        """SUBSUMES relation을 발견하면 child condition의 condition_source에 metadata 추가."""
        for r in relations:
            if r["type"].upper() != "SUBSUMES":
                continue
            parent_id = r.get("arg1_id")
            child_id = r.get("arg2_id")
            if not parent_id or not child_id:
                continue
            parent_entity = entity_dict.get(parent_id)
            if not parent_entity:
                continue
            parent_text = get_entity_text(parent_entity)
            # child entity가 변환된 condition을 찾아 metadata 추가
            for cond in condition_list:
                src_ids = cond.get("condition_source", {}).get("source_entity_ids", [])
                if child_id in src_ids:
                    cond["condition_source"]["subsumes_parent"] = parent_text
                    cond["condition_source"]["subsumes_relation"] = "SUBSUMES"

    # ─────────────────────────────────────────────────────────────
    # 전체 conversion_status 결정
    # ─────────────────────────────────────────────────────────────
    def _compute_overall_status(self, condition_list):
        if not condition_list:
            return "unconverted"
        statuses = [c.get("conversion_status", "unconverted") for c in condition_list]
        if all(s == "converted" for s in statuses):
            return "converted"
        if any(s == "converted" for s in statuses):
            return "partial"
        if all(s == "evidence_only" for s in statuses):
            return "evidence_only"
        return "partial"


# ════════════════════════════════════════════════════════════════
# 4. Validation (Rule check)
# ════════════════════════════════════════════════════════════════

def validate_condition(cond):
    """Method 스키마 정합성 검증. 위반 시 issue 리스트 반환."""
    issues = []
    ctype = cond.get("type")
    # required field check
    if ctype == "numeric_threshold":
        for f in ("variable", "comparator"):
            if not cond.get(f):
                issues.append(f"numeric_threshold missing required field: {f}")
        if cond.get("comparator") not in METHOD_COMPARATORS:
            issues.append(f"numeric_threshold invalid comparator: {cond.get('comparator')}")
    elif ctype == "categorical_state":
        for f in ("variable", "value"):
            if not cond.get(f):
                issues.append(f"categorical_state missing required field: {f}")
    elif ctype == "medication_history":
        for f in ("drug", "status"):
            if not cond.get(f):
                issues.append(f"medication_history missing required field: {f}")
        if cond.get("status") and cond["status"] not in METHOD_MED_STATUS:
            issues.append(f"medication_history invalid status: {cond['status']}")
    elif ctype == "temporal_condition":
        for f in ("event", "anchor", "comparator"):
            if f not in cond:
                issues.append(f"temporal_condition missing required field: {f}")

    # allowed value check
    if cond.get("clinical_status") and cond["clinical_status"] not in METHOD_CLINICAL_STATUS:
        issues.append(f"invalid clinical_status: {cond['clinical_status']}")
    if cond.get("verification_status") and cond["verification_status"] not in METHOD_VERIFICATION_STATUS:
        issues.append(f"invalid verification_status: {cond['verification_status']}")
    if cond.get("severity") and cond["severity"] not in METHOD_SEVERITY:
        issues.append(f"invalid severity: {cond['severity']}")
    if cond.get("temporal_relation") and cond["temporal_relation"] not in METHOD_TEMPORAL_RELATIONS:
        issues.append(f"invalid temporal_relation: {cond['temporal_relation']}")

    # conversion status check
    cs = cond.get("conversion_status")
    if cs not in {"converted", "partial", "unconverted", "evidence_only"}:
        issues.append(f"invalid conversion_status: {cs}")

    # source traceability check
    src = cond.get("condition_source", {})
    if "source_entity_ids" not in src:
        issues.append("missing source_entity_ids")

    return issues


# ════════════════════════════════════════════════════════════════
# 5. 메인 실행
# ════════════════════════════════════════════════════════════════

def main():
    OUTPUT_FILE = "chia_to_ours_quadruples.json"
    STATS_FILE = "chia_to_ours_stats.txt"

    print("=" * 70)
    print("Chia → Ours Translation Pipeline")
    print("=" * 70)

    # ─── 1. 데이터 로드 ──────────────────────────────────────────
    print("\n[1/4] Chia 데이터셋 로드 중...")
    try:
        ds = load_dataset("bigbio/chia", "chia_without_scope_fixed_source")
    except TypeError:
        # datasets 3.x: trust_remote_code 인자 시도
        ds = load_dataset("bigbio/chia", "chia_without_scope_fixed_source",
                          trust_remote_code=True)
    train = ds["train"]
    print(f"   ✓ 로드 완료: {len(train)}개 record")

    # ─── 2. 변환 ─────────────────────────────────────────────────
    print(f"\n[2/4] 변환 중...")
    converter = ChiaToOursConverter()
    quadruples = []
    validation_issues = []
    record_count = 0

    for idx, row in enumerate(train):
        q_list = converter.convert_document(row)
        quadruples.extend(q_list)
        record_count += 1
        # validation
        for q in q_list:
            for cond in q["condition_list"]:
                issues = validate_condition(cond)
                if issues:
                    validation_issues.append({
                        "record_id": q.get("id"),
                        "condition_type": cond.get("type"),
                        "issues": issues,
                    })
        if (idx + 1) % 500 == 0:
            print(f"   {idx+1}/{len(train)} 처리됨...")

    print(f"   ✓ 변환 완료: {record_count}개 record → {len(quadruples)}개 quadruple 생성")

    # ─── 3. 통계 출력 ────────────────────────────────────────────
    print(f"\n[3/4] 통계 작성 중...")
    stats_lines = []
    stats_lines.append("=" * 70)
    stats_lines.append("Chia → Ours 변환 통계")
    stats_lines.append("=" * 70)
    stats_lines.append(f"\n전체 record: {record_count}개 → quadruple: {len(quadruples)}개")
    stats_lines.append(f"  평균 문장/record: {len(quadruples)/max(record_count,1):.1f}개")

    # text_type 분포
    tt_dist = Counter(q["text_type"] for q in quadruples)
    stats_lines.append(f"\n[text_type 분포]")
    for tt, cnt in tt_dist.most_common():
        stats_lines.append(f"  {tt}: {cnt} ({cnt/len(quadruples)*100:.1f}%)")

    # triple relation 분포
    rel_dist = Counter(q["triple"]["relation"] for q in quadruples)
    stats_lines.append(f"\n[triple relation 분포]")
    for rel, cnt in rel_dist.most_common():
        stats_lines.append(f"  {rel}: {cnt} ({cnt/len(quadruples)*100:.1f}%)")

    # overall conversion_status 분포
    cs_dist = Counter(q["conversion_status"] for q in quadruples)
    stats_lines.append(f"\n[Overall conversion_status 분포]")
    for cs, cnt in cs_dist.most_common():
        stats_lines.append(f"  {cs}: {cnt} ({cnt/len(quadruples)*100:.1f}%)")

    # condition 단위 통계
    all_conditions = [c for q in quadruples for c in q["condition_list"]]
    stats_lines.append(f"\n[condition 단위 통계]")
    stats_lines.append(f"  총 condition: {len(all_conditions)}개")
    type_dist = Counter(c["type"] for c in all_conditions)
    for t, cnt in type_dist.most_common():
        stats_lines.append(f"  {t}: {cnt} ({cnt/max(len(all_conditions),1)*100:.1f}%)")

    cond_cs_dist = Counter(c.get("conversion_status", "unknown") for c in all_conditions)
    stats_lines.append(f"\n[condition별 conversion_status 분포]")
    for cs, cnt in cond_cs_dist.most_common():
        stats_lines.append(f"  {cs}: {cnt} ({cnt/max(len(all_conditions),1)*100:.1f}%)")

    # validation 결과
    stats_lines.append(f"\n[Validation 결과]")
    stats_lines.append(f"  Issue 건수: {len(validation_issues)}")
    if validation_issues:
        issue_types = Counter()
        for vi in validation_issues:
            for issue in vi["issues"]:
                issue_types[issue.split(":")[0]] += 1
        for it, cnt in issue_types.most_common(10):
            stats_lines.append(f"    {it}: {cnt}")

    stats_text = "\n".join(stats_lines)
    print(stats_text)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        f.write(stats_text)
    print(f"\n   ✓ 통계 저장: {STATS_FILE}")

    # ─── 4. JSON 출력 ────────────────────────────────────────────
    print(f"\n[4/4] JSON 출력 중...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(quadruples, f, ensure_ascii=False, indent=2)
    print(f"   ✓ 저장 완료: {OUTPUT_FILE}")

    print(f"\n{'=' * 70}")
    print(f"파이프라인 완료")
    print(f"  - quadruples → {OUTPUT_FILE}")
    print(f"  - statistics → {STATS_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()