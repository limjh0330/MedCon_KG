"""
Entity <-> Condition cross-duplication analyzer.

Question: 동일한 임상 개념이 entity(head_name/tail_name)로도 표현되고
condition(value / evidence_text / evidence_texts)으로도 동시에 표현되는가?

검사 축:
  (1) GLOBAL: 모든 entity 이름의 집합 vs 모든 condition 텍스트의 집합 (정확 일치)
  (2) GLOBAL substring: 한쪽이 다른 쪽의 부분문자열 (대용량이라 토큰 빈도가 높은 항목으로 제한)
  (3) PER-TRIPLE: 같은 triple 내부에서 그 triple 자신의 head/tail이 condition 값/텍스트에 들어있는가
  (4) condition 측에서 사용되는 'variable' 라벨이 entity 이름과 겹치는가
  (5) tail_name 이 condition_source.evidence_texts 와 겹치는가 (가이드라인 문장이 그래프 노드로도 재등장하는지)
"""
import io, sys, json, re
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATH = Path("output/stage3_condition_augmented_triples.json")

# ---- Load + patch known corruption ------------------------------
with PATH.open("r", encoding="utf-8") as f:
    raw = f.read()
raw = raw.replace('    {"O"\n', '    {\n')
data = json.loads(raw)
del raw
triples = data["triples"]
N = len(triples)
print(f"Loaded triples: {N:,}")

def norm(s):
    """소문자 + 양끝 공백 제거 + 다중공백 1개로"""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def pct(x, y):
    return f"{(x/y*100):.2f}%" if y else "n/a"

# ---- 1) Entity 집합 빌드 -----------------------------------------
print("\n[1/5] Building entity / condition string pools ...")
head_names = [t.get("head_name", "") for t in triples]
tail_names = [t.get("tail_name", "") for t in triples]

head_name_norm  = {norm(s) for s in head_names if s}
tail_name_norm  = {norm(s) for s in tail_names if s}
all_entity_norm = head_name_norm | tail_name_norm
print(f"  unique norm(head_name) : {len(head_name_norm):,}")
print(f"  unique norm(tail_name) : {len(tail_name_norm):,}")
print(f"  unique norm(all entity): {len(all_entity_norm):,}")

# ---- 2) Condition 텍스트 풀 빌드 ---------------------------------
cond_value_pool        = []   # condition.value (string화)
cond_evidence_pool     = []   # condition.evidence_text
cond_variable_pool     = []   # condition.variable
src_evidence_pool      = []   # condition_source.evidence_texts (flat)
cond_value_to_triples  = defaultdict(set)   # cond.value -> { triple-keys that use it }
cond_evid_to_triples   = defaultdict(set)
src_evid_to_triples    = defaultdict(set)

triples_with_cond = [t for t in triples if t.get("has_conditions")]
for t in triples_with_cond:
    tkey = (t.get("head_name",""), t.get("relation",""), t.get("tail_name",""))
    for c in (t.get("conditions") or []):
        if not isinstance(c, dict):
            continue
        v = c.get("value")
        if v is not None:
            v_str = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
            cond_value_pool.append(v_str)
            cond_value_to_triples[norm(v_str)].add(tkey)
        if c.get("evidence_text"):
            cond_evidence_pool.append(c["evidence_text"])
            cond_evid_to_triples[norm(c["evidence_text"])].add(tkey)
        if c.get("variable"):
            cond_variable_pool.append(c["variable"])
    src = t.get("condition_source") or {}
    for et in (src.get("evidence_texts") or []):
        src_evidence_pool.append(et)
        src_evid_to_triples[norm(et)].add(tkey)

cond_value_norm    = {norm(s) for s in cond_value_pool if s}
cond_evidence_norm = {norm(s) for s in cond_evidence_pool if s}
cond_variable_norm = {norm(s) for s in cond_variable_pool if s}
src_evidence_norm  = {norm(s) for s in src_evidence_pool if s}

print(f"  condition.value    pool  : total={len(cond_value_pool):,}  unique={len(cond_value_norm):,}")
print(f"  condition.evidence pool  : total={len(cond_evidence_pool):,}  unique={len(cond_evidence_norm):,}")
print(f"  condition.variable pool  : total={len(cond_variable_pool):,}  unique={len(cond_variable_norm):,}")
print(f"  cond_source.evidence_text pool : total={len(src_evidence_pool):,}  unique={len(src_evidence_norm):,}")

# ===================================================================
# (A) GLOBAL EXACT-MATCH OVERLAP
# ===================================================================
print("\n" + "=" * 78)
print("[A] GLOBAL exact-string overlap (case/space normalized)")
print("=" * 78)

def report_overlap(label, A, B, name_a="entity", name_b="cond"):
    inter = A & B
    print(f"\n  {label}")
    print(f"    {name_a:<8} unique : {len(A):>10,}")
    print(f"    {name_b:<8} unique : {len(B):>10,}")
    print(f"    intersection      : {len(inter):>10,}")
    if A: print(f"    overlap of {name_a:<6}: {pct(len(inter), len(A))}")
    if B: print(f"    overlap of {name_b:<6}: {pct(len(inter), len(B))}")
    return inter

ov1 = report_overlap("entity (head ∪ tail)  ∩  condition.value",
                     all_entity_norm, cond_value_norm, "entity", "cond.val")
ov2 = report_overlap("entity (head ∪ tail)  ∩  condition.evidence_text",
                     all_entity_norm, cond_evidence_norm, "entity", "cond.evt")
ov3 = report_overlap("entity (head ∪ tail)  ∩  condition.variable",
                     all_entity_norm, cond_variable_norm, "entity", "cond.var")
ov4 = report_overlap("entity (head ∪ tail)  ∩  cond_source.evidence_texts",
                     all_entity_norm, src_evidence_norm, "entity", "src.evt")

ov_head_v = report_overlap("head_name  ∩  condition.value",
                           head_name_norm, cond_value_norm, "head", "cond.val")
ov_tail_v = report_overlap("tail_name  ∩  condition.value",
                           tail_name_norm, cond_value_norm, "tail", "cond.val")

# 예시 출력 (가장 흥미로운 ov1 = entity ∩ cond.value)
print("\n  -- Sample overlapping strings (entity ∩ condition.value), up to 30 --")
for s in sorted(ov1)[:30]:
    print(f"    {s!r}")

# 어떤 entity가 가장 자주 condition으로도 등장하는지
print("\n  -- Most-impactful overlaps: entity-string also used as condition.value --")
ent_all_counter = Counter(norm(s) for s in head_names + tail_names if s)
val_counter     = Counter(norm(s) for s in cond_value_pool if s)
ranked = sorted(
    ((s, ent_all_counter.get(s, 0), val_counter.get(s, 0))
     for s in ov1),
    key=lambda x: -(x[1] + x[2]),
)[:15]
print(f"    {'string':<60} {'as_entity':>10} {'as_cond.val':>12}")
for s, ec, vc in ranked:
    label = (s[:57] + "...") if len(s) > 60 else s
    print(f"    {label:<60} {ec:>10,} {vc:>12,}")

# ===================================================================
# (B) PER-TRIPLE OVERLAP — most direct redundancy
# ===================================================================
print("\n" + "=" * 78)
print("[B] PER-TRIPLE overlap — same triple expresses same concept as both entity & condition")
print("=" * 78)

cnt_head_in_value      = 0
cnt_tail_in_value      = 0
cnt_head_in_evidence   = 0
cnt_tail_in_evidence   = 0
cnt_head_in_src_evt    = 0
cnt_tail_in_src_evt    = 0

# substring (head/tail이 evidence_text 안에 부분문자열로 포함되는 더 유연한 매칭)
cnt_head_substr_evt    = 0
cnt_tail_substr_evt    = 0

examples = []   # (kind, head, rel, tail, snippet)

for t in triples_with_cond:
    h_raw = t.get("head_name","")
    tl_raw = t.get("tail_name","")
    h, tl = norm(h_raw), norm(tl_raw)
    rel = t.get("relation","")
    cond_values, cond_evts = [], []
    for c in (t.get("conditions") or []):
        if not isinstance(c, dict): continue
        v = c.get("value")
        if v is not None:
            v_str = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
            cond_values.append(norm(v_str))
        if c.get("evidence_text"):
            cond_evts.append(norm(c["evidence_text"]))
    src_evts = [norm(s) for s in (t.get("condition_source") or {}).get("evidence_texts") or []]

    if h and h in cond_values:
        cnt_head_in_value += 1
        if len(examples) < 6:
            examples.append(("HEAD==cond.value", h_raw, rel, tl_raw, h))
    if tl and tl in cond_values:
        cnt_tail_in_value += 1
        if len(examples) < 12:
            examples.append(("TAIL==cond.value", h_raw, rel, tl_raw, tl))
    if h and h in cond_evts:
        cnt_head_in_evidence += 1
    if tl and tl in cond_evts:
        cnt_tail_in_evidence += 1
        if len(examples) < 18:
            examples.append(("TAIL==cond.evidence", h_raw, rel, tl_raw, tl))
    if h and h in src_evts:
        cnt_head_in_src_evt += 1
    if tl and tl in src_evts:
        cnt_tail_in_src_evt += 1

    # substring (length>=4 짧은 단어 노이즈 회피)
    if h and len(h) >= 4:
        for ev in cond_evts:
            if h in ev:
                cnt_head_substr_evt += 1
                break
    if tl and len(tl) >= 4:
        for ev in cond_evts:
            if tl in ev:
                cnt_tail_substr_evt += 1
                break

print(f"\n  triples_with_conditions : {len(triples_with_cond):,}")
print(f"\n  EXACT match (after lower/space-normalize):")
print(f"    head_name appears as condition.value         : {cnt_head_in_value:,}  ({pct(cnt_head_in_value, len(triples_with_cond))})")
print(f"    tail_name appears as condition.value         : {cnt_tail_in_value:,}  ({pct(cnt_tail_in_value, len(triples_with_cond))})")
print(f"    head_name appears as condition.evidence_text : {cnt_head_in_evidence:,}  ({pct(cnt_head_in_evidence, len(triples_with_cond))})")
print(f"    tail_name appears as condition.evidence_text : {cnt_tail_in_evidence:,}  ({pct(cnt_tail_in_evidence, len(triples_with_cond))})")
print(f"    head_name appears in cond_source.evidence_texts: {cnt_head_in_src_evt:,}")
print(f"    tail_name appears in cond_source.evidence_texts: {cnt_tail_in_src_evt:,}")

print(f"\n  SUBSTRING match (head/tail >=4 chars, contained in evidence_text):")
print(f"    head_name SUBSTRING-of condition.evidence_text : {cnt_head_substr_evt:,}  ({pct(cnt_head_substr_evt, len(triples_with_cond))})")
print(f"    tail_name SUBSTRING-of condition.evidence_text : {cnt_tail_substr_evt:,}  ({pct(cnt_tail_substr_evt, len(triples_with_cond))})")

print("\n  -- Examples (entity == condition value/text within the same triple) --")
for kind, h, r, tl, snip in examples:
    print(f"    [{kind}] {h!r} --{r}--> {tl!r}")
    print(f"        condition snippet: {snip!r}")

# ===================================================================
# (C) condition.variable 라벨이 entity 이름과 겹치는가
# ===================================================================
print("\n" + "=" * 78)
print("[C] condition.variable overlap with entity names")
print("=" * 78)
report_overlap("entity (head ∪ tail) ∩ condition.variable",
               all_entity_norm, cond_variable_norm, "entity", "cond.var")
print("\n  variable values that ALSO appear as an entity name:")
for v in sorted(all_entity_norm & cond_variable_norm)[:30]:
    print(f"    {v!r}")

# ===================================================================
# (D) Aggregate summary
# ===================================================================
print("\n" + "=" * 78)
print("[D] SUMMARY -- entity-condition cross-duplication")
print("=" * 78)
print(f"  triples_with_conditions    : {len(triples_with_cond):,}")
print(f"  unique entity names        : {len(all_entity_norm):,}")
print(f"  unique condition.value     : {len(cond_value_norm):,}")
print(f"  unique condition.evidence  : {len(cond_evidence_norm):,}")
print()
print(f"  entity ∩ condition.value         (exact) : {len(ov1):,}  "
      f"=> covers {pct(len(ov1), len(cond_value_norm))} of unique condition.values")
print(f"  entity ∩ condition.evidence_text (exact) : {len(ov2):,}  "
      f"=> covers {pct(len(ov2), len(cond_evidence_norm))} of unique condition.evidences")
print(f"  entity ∩ condition.variable      (exact) : {len(ov3):,}")
print(f"  entity ∩ cond_source.evidence_texts(exact): {len(ov4):,}")
print()
print(f"  PER-TRIPLE same-concept redundancy:")
print(f"    head_name == any cond.value in same triple    : {cnt_head_in_value:,}")
print(f"    tail_name == any cond.value in same triple    : {cnt_tail_in_value:,}")
print(f"    head_name == any cond.evidence in same triple : {cnt_head_in_evidence:,}")
print(f"    tail_name == any cond.evidence in same triple : {cnt_tail_in_evidence:,}")
print(f"    head substring of cond.evidence (same triple) : {cnt_head_substr_evt:,}")
print(f"    tail substring of cond.evidence (same triple) : {cnt_tail_substr_evt:,}")

print("\nDONE.")
