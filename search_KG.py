"""
Search KG entities extracted from triples against input unique entities.
"""

import logging

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """Normalize text for exact string matching."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.casefold().split())


def _build_kg_entities(triples: list[dict]) -> list[dict]:
    """Build a unique KG entity list from triple head/tail items."""
    entity_map = {}

    for triple in triples:
        head_id = triple.get("head_cui", "")
        head_name = triple.get("head_name", "")
        if head_id or head_name:
            key = (head_id, head_name)
            entry = entity_map.setdefault(
                key,
                {
                    "entity_id": head_id,
                    "entity_name": head_name,
                    "entity_sources": set(),
                    "triple_count": 0,
                },
            )
            entry["entity_sources"].add("head")
            entry["triple_count"] += 1

        tail_id = triple.get("tail_id", "")
        tail_name = triple.get("tail_name", "")
        if tail_id or tail_name:
            key = (tail_id, tail_name)
            entry = entity_map.setdefault(
                key,
                {
                    "entity_id": tail_id,
                    "entity_name": tail_name,
                    "entity_sources": set(),
                    "triple_count": 0,
                },
            )
            entry["entity_sources"].add("tail")
            entry["triple_count"] += 1

    kg_entities = []
    for entry in entity_map.values():
        kg_entities.append(
            {
                "entity_id": entry["entity_id"],
                "entity_name": entry["entity_name"],
                "entity_sources": sorted(entry["entity_sources"]),
                "triple_count": entry["triple_count"],
            }
        )
    return kg_entities


def _build_kg_index(kg_entities: list[dict]) -> dict:
    """Index KG entities by normalized entity name."""
    kg_index = {}

    for entity in kg_entities:
        key = _normalize_text(entity.get("entity_name", ""))
        if key:
            kg_index.setdefault(key, []).append(entity)

    return kg_index


def search_KG(kg_info: dict, input_info: dict) -> dict:
    """
    Compare KG triple entities with input unique entities.

    Args:
        kg_info: run_pipeline() result dict containing `triples`
        input_info: get_query_entity() result dict containing `unique_entities`
    """
    triples = kg_info.get("triples", [])
    input_entities = input_info.get("unique_entities", [])

    if not isinstance(triples, list):
        raise ValueError("kg_info['triples'] must be a list")
    if not isinstance(input_entities, list):
        raise ValueError("input_info['unique_entities'] must be a list")

    kg_entities = _build_kg_entities(triples)
    kg_index = _build_kg_index(kg_entities)

    matched_results = []
    unmatched_results = []

    for input_entity in input_entities:
        if not isinstance(input_entity, dict):
            continue

        normalized_key = _normalize_text(
            input_entity.get("normalized_form") or input_entity.get("surface_form", "")
        )
        surface_key = _normalize_text(input_entity.get("surface_form", ""))

        matched_entities = []
        match_type = ""

        if normalized_key and normalized_key in kg_index:
            matched_entities = kg_index[normalized_key]
            match_type = "normalized_form"
        elif surface_key and surface_key in kg_index:
            matched_entities = kg_index[surface_key]
            match_type = "surface_form"

        result = {
            "input_entity": input_entity,
        }

        if matched_entities:
            result["match_type"] = match_type
            result["matched_entities"] = matched_entities
            matched_results.append(result)
        else:
            unmatched_results.append(result)

    logger.info(
        "KG entity search complete: %s/%s input unique entities matched",
        len(matched_results),
        len(input_entities),
    )

    return {
        "total_triples": len(triples),
        "total_kg_entities": len(kg_entities),
        "total_input_unique_entities": len(input_entities),
        "matched_input_entities": len(matched_results),
        "unmatched_input_entities": len(unmatched_results),
        "match_rate": round(
            len(matched_results) / max(len(input_entities), 1) * 100, 1
        ),
        "matches": matched_results,
        "unmatched": unmatched_results,
    }
