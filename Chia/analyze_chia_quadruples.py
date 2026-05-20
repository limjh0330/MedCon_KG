"""
chia_to_ours_quadruples.json 분석 스크립트 (v3)
================================================

수정된 converter의 개선안 효과를 추적하는 섹션(J)을 포함.
v3: J-2에 subtype별 variable 샘플, J-3에 current drug 오탐 검증 리스트 추가.

분석 항목:
  A. 전체 규모 요약
  B. text_type / triple relation 분포
  C. condition_type별 분포 및 필드 완성도
  D. conversion_status 분포 (quadruple-level / condition-level)
  E. condition_logic 분포
  F. 변환 품질 심층 분석 (partial/unconverted 원인)
  G. entity 흡수 및 추적성 분석
  H. 문서(document_id)별 요약
  I. 샘플 출력 (type별 대표 예시)
  J. 개선안 효과 분석 (INTERVAL 분리 / subtype 확장 / med status 추론)

사용법:
  python analyze_chia_quadruples.py [json_path]
  (기본값: chia_to_ours_quadruples.json)
"""

import json
import sys
from collections import Counter, defaultdict


# ═══════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def section(title):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def bar(label, count, total, width=30):
    pct = count / max(total, 1) * 100
    filled = int(pct / 100 * width)
    return f"  {label:<30s} {count:>6d}  ({pct:5.1f}%) {'█' * filled}{'░' * (width - filled)}"


# ═══════════════════════════════════════════════════════
# A. 전체 규모 요약
# ═══════════════════════════════════════════════════════

def analysis_a_overview(quads):
    section("A. 전체 규모 요약")

    n_quads = len(quads)
    doc_ids = set(q["document_id"] for q in quads)
    all_conds = [c for q in quads for c in q["condition_list"]]

    doc_sent_counts = Counter(q["document_id"] for q in quads)
    avg_sent = n_quads / max(len(doc_ids), 1)
    max_sent_doc = doc_sent_counts.most_common(1)[0] if doc_sent_counts else ("N/A", 0)

    cond_counts = [len(q["condition_list"]) for q in quads]
    avg_cond = sum(cond_counts) / max(n_quads, 1)
    empty_quads = sum(1 for c in cond_counts if c == 0)

    print(f"  총 quadruple 수        : {n_quads:,}")
    print(f"  원본 document 수       : {len(doc_ids):,}")
    print(f"  총 condition 수        : {len(all_conds):,}")
    print(f"  평균 문장/document     : {avg_sent:.1f}")
    print(f"  최다 문장 document     : {max_sent_doc[0]} ({max_sent_doc[1]}문장)")
    print(f"  평균 condition/quadruple: {avg_cond:.1f}")
    print(f"  condition 0개 quadruple: {empty_quads} ({empty_quads/max(n_quads,1)*100:.1f}%)")

    print(f"\n  [quadruple당 condition 수 분포]")
    cond_hist = Counter(cond_counts)
    for k in sorted(cond_hist.keys())[:10]:
        print(bar(f"  {k}개 condition", cond_hist[k], n_quads))

    return all_conds


# ═══════════════════════════════════════════════════════
# B. text_type / triple relation 분포
# ═══════════════════════════════════════════════════════

def analysis_b_text_type_and_triple(quads):
    section("B. text_type 및 triple relation 분포")

    n = len(quads)

    tt_dist = Counter(q["text_type"] for q in quads)
    print("  [text_type]")
    for tt, cnt in tt_dist.most_common():
        print(bar(tt, cnt, n))

    rel_dist = Counter(q["triple"]["relation"] for q in quads)
    print(f"\n  [triple relation]")
    for rel, cnt in rel_dist.most_common():
        print(bar(rel, cnt, n))

    exc_tails = [q["triple"]["tail"] for q in quads if q["text_type"] == "exclusion"]
    if exc_tails:
        tail_dist = Counter(exc_tails)
        print(f"\n  [exclusion tail 상위 10개]")
        for tail, cnt in tail_dist.most_common(10):
            print(f"    {tail:<45s}  {cnt:>4d}")


# ═══════════════════════════════════════════════════════
# C. condition type별 분포 및 필드 완성도
# ═══════════════════════════════════════════════════════

def analysis_c_condition_types(all_conds):
    section("C. condition type별 분포 및 필드 완성도")

    n = len(all_conds)
    type_dist = Counter(c["type"] for c in all_conds)

    print("  [condition type 분포]")
    for t, cnt in type_dist.most_common():
        print(bar(t, cnt, n))

    type_fields = {
        "numeric_threshold":  ["variable", "comparator", "value", "unit", "subtype",
                                "value_min", "value_max", "inclusive_min", "inclusive_max"],
        "categorical_state":  ["variable", "value", "subtype", "clinical_status",
                                "severity", "verification_status"],
        "medication_history": ["drug", "status", "subtype", "dose", "unit", "frequency"],
        "temporal_condition": ["event", "anchor", "comparator", "interval",
                                "interval_unit", "temporal_relation"],
    }

    for ctype, fields in type_fields.items():
        conds_of_type = [c for c in all_conds if c["type"] == ctype]
        if not conds_of_type:
            continue
        cnt = len(conds_of_type)
        print(f"\n  [{ctype}] — {cnt}개")
        for f in fields:
            filled = sum(1 for c in conds_of_type if c.get(f) not in (None, "", "unknown"))
            pct = filled / cnt * 100
            status = "✓" if pct > 80 else ("△" if pct > 30 else "✗")
            print(f"    {status} {f:<22s}  {filled:>5d}/{cnt}  ({pct:5.1f}%)")

    cat_conds = [c for c in all_conds if c["type"] == "categorical_state"]
    if cat_conds:
        sub_dist = Counter(c.get("subtype", "N/A") for c in cat_conds)
        print(f"\n  [categorical_state subtype 분포]")
        for s, cnt in sub_dist.most_common():
            print(bar(s, cnt, len(cat_conds)))

    med_conds = [c for c in all_conds if c["type"] == "medication_history"]
    if med_conds:
        status_dist = Counter(c.get("status", "N/A") for c in med_conds)
        print(f"\n  [medication_history status 분포]")
        for s, cnt in status_dist.most_common():
            print(bar(s, cnt, len(med_conds)))

    nt_conds = [c for c in all_conds if c["type"] == "numeric_threshold"]
    if nt_conds:
        nt_sub = Counter(c.get("subtype") or "(미설정)" for c in nt_conds)
        print(f"\n  [numeric_threshold subtype 분포]")
        for s, cnt in nt_sub.most_common():
            print(bar(s, cnt, len(nt_conds)))

    if med_conds:
        med_sub = Counter(c.get("subtype") or "(미설정)" for c in med_conds)
        print(f"\n  [medication_history subtype 분포]")
        for s, cnt in med_sub.most_common():
            print(bar(s, cnt, len(med_conds)))


# ═══════════════════════════════════════════════════════
# D. conversion_status 분포
# ═══════════════════════════════════════════════════════

def analysis_d_conversion_status(quads, all_conds):
    section("D. conversion_status 분포")

    n_q = len(quads)
    n_c = len(all_conds)

    q_cs = Counter(q["conversion_status"] for q in quads)
    print("  [quadruple-level]")
    for cs, cnt in q_cs.most_common():
        print(bar(cs, cnt, n_q))

    c_cs = Counter(c.get("conversion_status", "unknown") for c in all_conds)
    print(f"\n  [condition-level]")
    for cs, cnt in c_cs.most_common():
        print(bar(cs, cnt, n_c))

    print(f"\n  [type × conversion_status 교차표]")
    types = sorted(set(c["type"] for c in all_conds))
    statuses = ["converted", "partial", "unconverted", "evidence_only"]
    header = f"  {'type':<25s}" + "".join(f"{s:>14s}" for s in statuses)
    print(header)
    print("  " + "─" * (25 + 14 * len(statuses)))
    for t in types:
        row_conds = [c for c in all_conds if c["type"] == t]
        row_cs = Counter(c.get("conversion_status", "unknown") for c in row_conds)
        cells = "".join(f"{row_cs.get(s, 0):>14d}" for s in statuses)
        print(f"  {t:<25s}{cells}")


# ═══════════════════════════════════════════════════════
# E. condition_logic 분포
# ═══════════════════════════════════════════════════════

def analysis_e_logic(quads):
    section("E. condition_logic 분포")

    n = len(quads)
    logic_dist = Counter(q.get("condition_logic", "N/A") for q in quads)
    for lg, cnt in logic_dist.most_common():
        print(bar(lg, cnt, n))

    multi = [q for q in quads if len(q["condition_list"]) >= 2]
    if multi:
        multi_logic = Counter(q.get("condition_logic", "N/A") for q in multi)
        print(f"\n  [condition ≥ 2개인 quadruple ({len(multi)}개) 에서의 logic]")
        for lg, cnt in multi_logic.most_common():
            print(bar(lg, cnt, len(multi)))


# ═══════════════════════════════════════════════════════
# F. 변환 품질 심층 분석
# ═══════════════════════════════════════════════════════

def analysis_f_quality_deep(quads, all_conds):
    section("F. 변환 품질 심층 분석")

    problem_conds = [c for c in all_conds
                     if c.get("conversion_status") in ("partial", "unconverted")]
    print(f"  partial + unconverted condition 수: {len(problem_conds)}")

    if problem_conds:
        prob_type = Counter(c["type"] for c in problem_conds)
        print(f"\n  [문제 condition의 type 분포]")
        for t, cnt in prob_type.most_common():
            print(bar(t, cnt, len(problem_conds)))

        evidence_words = Counter()
        for c in problem_conds:
            et = c.get("evidence_text", "")
            for word in et.lower().split():
                if len(word) > 3:
                    evidence_words[word] += 1
        if evidence_words:
            print(f"\n  [partial/unconverted evidence_text 빈출 키워드 상위 15]")
            for w, cnt in evidence_words.most_common(15):
                print(f"    {w:<25s}  {cnt:>4d}")

    eo_conds = [c for c in all_conds if c.get("conversion_status") == "evidence_only"]
    if eo_conds:
        print(f"\n  evidence_only condition 수: {len(eo_conds)}")
        eo_types = Counter(c["type"] for c in eo_conds)
        for t, cnt in eo_types.most_common():
            print(f"    {t}: {cnt}")


# ═══════════════════════════════════════════════════════
# G. entity 흡수 및 추적성 분석
# ═══════════════════════════════════════════════════════

def analysis_g_traceability(all_conds):
    section("G. entity 흡수 및 추적성 분석")

    eid_counts = [len(c.get("condition_source", {}).get("source_entity_ids", []))
                  for c in all_conds]
    eid_hist = Counter(eid_counts)
    print("  [condition당 흡수 entity 수 분포]")
    for k in sorted(eid_hist.keys())[:8]:
        print(bar(f"{k}개 entity 흡수", eid_hist[k], len(all_conds)))

    subsumes_conds = [c for c in all_conds
                      if c.get("condition_source", {}).get("subsumes_parent")]
    print(f"\n  SUBSUMES metadata 보유 condition: {len(subsumes_conds)}")

    missing_src = sum(1 for c in all_conds if not c.get("condition_source"))
    missing_eids = sum(1 for c in all_conds
                       if "source_entity_ids" not in c.get("condition_source", {}))
    print(f"  condition_source 누락: {missing_src}")
    print(f"  source_entity_ids 누락: {missing_eids}")


# ═══════════════════════════════════════════════════════
# H. 문서(document_id)별 요약
# ═══════════════════════════════════════════════════════

def analysis_h_per_document(quads):
    section("H. 문서별 요약 (상위 10)")

    doc_groups = defaultdict(list)
    for q in quads:
        doc_groups[q["document_id"]].append(q)

    doc_stats = []
    for doc_id, qs in doc_groups.items():
        n_conds = sum(len(q["condition_list"]) for q in qs)
        n_converted = sum(1 for q in qs if q["conversion_status"] == "converted")
        doc_stats.append({
            "doc_id": doc_id,
            "sentences": len(qs),
            "conditions": n_conds,
            "converted_rate": n_converted / max(len(qs), 1) * 100,
        })

    doc_stats.sort(key=lambda x: x["sentences"], reverse=True)

    print(f"  {'document_id':<35s} {'문장':>5s} {'조건':>5s} {'변환율':>7s}")
    print("  " + "─" * 55)
    for d in doc_stats[:10]:
        print(f"  {d['doc_id']:<35s} {d['sentences']:>5d} {d['conditions']:>5d} "
              f"{d['converted_rate']:>6.1f}%")

    rates = [d["converted_rate"] for d in doc_stats]
    print(f"\n  [문서별 변환율 분포]")
    bins = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 101)]
    labels = ["0-25%", "25-50%", "50-75%", "75-100%", "100%"]
    for (lo, hi), label in zip(bins, labels):
        cnt = sum(1 for r in rates if lo <= r < hi)
        print(bar(label, cnt, len(rates)))


# ═══════════════════════════════════════════════════════
# I. 샘플 출력 (type별 대표 예시)
# ═══════════════════════════════════════════════════════

def analysis_i_samples(quads, all_conds):
    section("I. type별 대표 condition 샘플 (converted 우선)")

    target_types = ["numeric_threshold", "categorical_state",
                    "medication_history", "temporal_condition"]

    for ctype in target_types:
        candidates = [c for c in all_conds
                      if c["type"] == ctype and c.get("conversion_status") == "converted"]
        if not candidates:
            candidates = [c for c in all_conds if c["type"] == ctype]
        if not candidates:
            print(f"\n  [{ctype}] — 해당 없음")
            continue

        sample = candidates[0]
        print(f"\n  [{ctype}]")
        skip_keys = {"condition_source"}
        for k, v in sample.items():
            if k in skip_keys:
                print(f"    {k}: (생략 — guideline_id: {v.get('guideline_id', 'N/A')}, "
                      f"entity_ids: {len(v.get('source_entity_ids', []))}개)")
            else:
                print(f"    {k}: {v}")

    print(f"\n  [원문 ↔ condition 대조 샘플]")
    good = [q for q in quads
            if len(q["condition_list"]) >= 2
            and q["conversion_status"] == "converted"]
    if not good:
        good = [q for q in quads if len(q["condition_list"]) >= 2]
    if good:
        sample_q = good[0]
        print(f"    id           : {sample_q['id']}")
        print(f"    text_type    : {sample_q['text_type']}")
        print(f"    sentence_text: {sample_q['sentence_text'][:80]}...")
        print(f"    triple       : {sample_q['triple']}")
        print(f"    logic        : {sample_q['condition_logic']}")
        print(f"    status       : {sample_q['conversion_status']}")
        print(f"    conditions   : {len(sample_q['condition_list'])}개")
        for i, c in enumerate(sample_q["condition_list"]):
            print(f"      [{i+1}] type={c['type']}, "
                  f"status={c.get('conversion_status')}, "
                  f"evidence={c.get('evidence_text', '')[:40]}")


# ═══════════════════════════════════════════════════════
# J. 개선안 효과 분석 (3가지 개선안 추적)
# ═══════════════════════════════════════════════════════

def analysis_j_improvement_tracking(quads, all_conds):
    section("J. 개선안 효과 분석")

    # ─── J-1: INTERVAL temporal 분리 생성 효과 ──────────────────
    print("  [J-1] INTERVAL → temporal_condition 분리 생성 효과")
    print("  " + "─" * 55)

    tc_conds = [c for c in all_conds if c["type"] == "temporal_condition"]
    cat_conds = [c for c in all_conds if c["type"] == "categorical_state"]
    n_tc = len(tc_conds)

    # source_entity_ids가 1개 = 분리 생성 추정 (Temporal entity 단독)
    spawned_tc = [c for c in tc_conds
                  if len(c.get("condition_source", {}).get("source_entity_ids", [])) == 1]
    standalone_tc = [c for c in tc_conds
                     if len(c.get("condition_source", {}).get("source_entity_ids", [])) > 1]

    print(f"  총 temporal_condition: {n_tc}")
    print(f"    분리 생성 추정 (entity 1개): {len(spawned_tc)}")
    print(f"    기존 standalone (entity 2+개): {len(standalone_tc)}")

    if spawned_tc:
        sp_converted = sum(1 for c in spawned_tc
                           if c.get("conversion_status") == "converted")
        print(f"    분리 생성 중 converted: {sp_converted}/{len(spawned_tc)} "
              f"({sp_converted/len(spawned_tc)*100:.1f}%)")

    cat_partial = sum(1 for c in cat_conds
                      if c.get("conversion_status") == "partial")
    cat_converted = sum(1 for c in cat_conds
                        if c.get("conversion_status") == "converted")
    cat_total = len(cat_conds)
    print(f"\n  categorical_state 변환 현황:")
    print(f"    converted : {cat_converted:>6d} ({cat_converted/max(cat_total,1)*100:.1f}%)")
    print(f"    partial   : {cat_partial:>6d} ({cat_partial/max(cat_total,1)*100:.1f}%)")

    cat_partial_conds = [c for c in cat_conds
                         if c.get("conversion_status") == "partial"]
    temporal_kw_in_partial = 0
    for c in cat_partial_conds:
        et = (c.get("evidence_text", "") or "").lower()
        if any(kw in et for kw in ["within", "months", "weeks", "days", "years",
                                     "prior to", "before", "after"]):
            temporal_kw_in_partial += 1
    print(f"    partial 중 시간 키워드 잔존: {temporal_kw_in_partial} "
          f"({temporal_kw_in_partial/max(len(cat_partial_conds),1)*100:.1f}%)")

    # ─── J-2: numeric_threshold subtype 확장 효과 ──────────────
    print(f"\n  [J-2] numeric_threshold subtype 확장 효과")
    print("  " + "─" * 55)

    nt_conds = [c for c in all_conds if c["type"] == "numeric_threshold"]
    if nt_conds:
        n_nt = len(nt_conds)
        has_subtype = sum(1 for c in nt_conds if c.get("subtype"))
        no_subtype = n_nt - has_subtype
        print(f"  총 numeric_threshold: {n_nt}")
        print(f"    subtype 설정됨: {has_subtype} ({has_subtype/n_nt*100:.1f}%)")
        print(f"    subtype 미설정: {no_subtype} ({no_subtype/n_nt*100:.1f}%)")

        sub_dist = Counter(c.get("subtype") or "(미설정)" for c in nt_conds)
        print(f"\n  subtype별 분포:")
        for s, cnt in sub_dist.most_common():
            print(bar(s, cnt, n_nt))

        # subtype별 대표 variable 텍스트 (키워드 매칭 검증용)
        print(f"\n  subtype별 variable 샘플 (매칭 검증):")
        for subtype in ["age", "vital_sign", "lab_value", "score", "imaging", "duration"]:
            sub_conds = [c for c in nt_conds if c.get("subtype") == subtype]
            if not sub_conds:
                continue
            var_dist = Counter(c.get("variable", "") for c in sub_conds)
            top3 = var_dist.most_common(3)
            examples = ", ".join(f"{v}({cnt})" for v, cnt in top3)
            print(f"    {subtype:<14s}: {examples}")

        no_sub = [c for c in nt_conds if not c.get("subtype")]
        if no_sub:
            var_dist = Counter(c.get("variable", "") for c in no_sub)
            print(f"\n  subtype 미설정 variable 상위 10개 (사전 확장 후보):")
            for v, cnt in var_dist.most_common(10):
                print(f"    {v:<45s}  {cnt:>4d}")

    if nt_conds:
        has_unit = sum(1 for c in nt_conds if c.get("unit"))
        print(f"\n  unit 완성도: {has_unit}/{len(nt_conds)} "
              f"({has_unit/len(nt_conds)*100:.1f}%)")

    # ─── J-3: medication_history status 추론 효과 ──────────────
    print(f"\n  [J-3] medication_history status 텍스트 추론 효과")
    print("  " + "─" * 55)

    med_conds = [c for c in all_conds if c["type"] == "medication_history"]
    if med_conds:
        n_med = len(med_conds)
        status_dist = Counter(c.get("status", "N/A") for c in med_conds)
        unknown_cnt = status_dist.get("unknown", 0)
        known_cnt = n_med - unknown_cnt

        print(f"  총 medication_history: {n_med}")
        print(f"    status 결정됨 (unknown 외): {known_cnt} ({known_cnt/n_med*100:.1f}%)")
        print(f"    status=unknown 잔존: {unknown_cnt} ({unknown_cnt/n_med*100:.1f}%)")

        print(f"\n  status별 분포:")
        for s, cnt in status_dist.most_common():
            print(bar(s, cnt, n_med))

        has_sub = sum(1 for c in med_conds if c.get("subtype"))
        print(f"\n  subtype 설정 비율: {has_sub}/{n_med} ({has_sub/n_med*100:.1f}%)")

        med_converted = sum(1 for c in med_conds
                            if c.get("conversion_status") == "converted")
        med_partial = sum(1 for c in med_conds
                          if c.get("conversion_status") == "partial")
        print(f"  conversion_status:")
        print(f"    converted: {med_converted} ({med_converted/n_med*100:.1f}%)")
        print(f"    partial  : {med_partial} ({med_partial/n_med*100:.1f}%)")

        unknown_meds = [c for c in med_conds if c.get("status") == "unknown"]
        if unknown_meds:
            drug_dist = Counter(c.get("drug", "") for c in unknown_meds)
            print(f"\n  status=unknown 잔존 drug 상위 10개:")
            for d, cnt in drug_dist.most_common(10):
                print(f"    {d:<45s}  {cnt:>4d}")

        # status=current drug 상위 10개 (오탐 검증용)
        current_meds = [c for c in med_conds if c.get("status") == "current"]
        if current_meds:
            cur_drug_dist = Counter(c.get("drug", "") for c in current_meds)
            print(f"\n  status=current drug 상위 10개 (오탐 검증):")
            for d, cnt in cur_drug_dist.most_common(10):
                print(f"    {d:<45s}  {cnt:>4d}")

    # ─── J-종합 ────────────────────────────────────────────────
    print(f"\n  [J-종합] 전체 변환 품질 요약")
    print("  " + "─" * 55)

    n_total = len(all_conds)
    cs_dist = Counter(c.get("conversion_status", "unknown") for c in all_conds)
    for cs in ["converted", "partial", "unconverted", "evidence_only"]:
        cnt = cs_dist.get(cs, 0)
        print(bar(cs, cnt, n_total))

    n_q = len(quads)
    q_cs = Counter(q["conversion_status"] for q in quads)
    print(f"\n  quadruple-level 변환율:")
    for cs in ["converted", "partial", "unconverted", "evidence_only"]:
        cnt = q_cs.get(cs, 0)
        print(bar(cs, cnt, n_q))


# ═══════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "chia_to_ours_quadruples.json"
    print(f"파일 로드 중: {path}")
    quads = load_data(path)
    print(f"로드 완료: {len(quads):,}개 quadruple\n")

    all_conds = analysis_a_overview(quads)
    analysis_b_text_type_and_triple(quads)
    analysis_c_condition_types(all_conds)
    analysis_d_conversion_status(quads, all_conds)
    analysis_e_logic(quads)
    analysis_f_quality_deep(quads, all_conds)
    analysis_g_traceability(all_conds)
    analysis_h_per_document(quads)
    analysis_i_samples(quads, all_conds)
    analysis_j_improvement_tracking(quads, all_conds)

    print(f"\n{'═' * 70}")
    print("  분석 완료")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()