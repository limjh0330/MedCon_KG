"""
UMLS semantic types grouped by semantic group.

Primary source: UMLS_KG/semantic_type_of_UMLS.json
Format:
{
    "GROUP_ABBR": {
        "group_name": str,
        "types": [(TUI, type_name, definition)],
    }
}
"""

import json
from pathlib import Path

import config


PROJECT_ROOT = Path(__file__).resolve().parent
SEMANTIC_TYPES_JSON = Path(config.SEMANTIC_TYPES_FILE)
if not SEMANTIC_TYPES_JSON.is_absolute():
    SEMANTIC_TYPES_JSON = PROJECT_ROOT / SEMANTIC_TYPES_JSON


def _clean_definition(definition: str) -> str:
    if definition is None:
        return ""

    cleaned = str(definition).strip()
    if cleaned.upper() == "NONE":
        return ""
    return cleaned


def _load_semantic_types_from_json(filepath: Path) -> dict:
    if not filepath.exists():
        raise FileNotFoundError(
            f"Semantic types JSON not found: {filepath}"
        )

    with open(filepath, "r", encoding="utf-8") as file:
        raw = json.load(file)

    semantic_type_rows = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and "semantic_type" in item:
                semantic_type_rows = item["semantic_type"]
                break
    elif isinstance(raw, dict):
        semantic_type_rows = raw.get("semantic_type", [])

    if not isinstance(semantic_type_rows, list):
        raise ValueError(
            f"Invalid semantic type payload in: {filepath}"
        )

    grouped = {}
    for row in semantic_type_rows:
        group_abbr = row["group_abbreviation"]
        group_name = row["semanticTypeGroup"]
        tui = row["TUI"]
        type_name = row["name"]
        definition = _clean_definition(row.get("definition", ""))

        if group_abbr not in grouped:
            grouped[group_abbr] = {
                "group_name": group_name,
                "types": [],
            }

        grouped[group_abbr]["types"].append((tui, type_name, definition))

    for group_data in grouped.values():
        group_data["types"].sort(key=lambda item: item[0])

    return dict(sorted(grouped.items()))


SEMANTIC_GROUPS_WITH_EXAMPLES = _load_semantic_types_from_json(
    SEMANTIC_TYPES_JSON
)


def build_prompt_semantic_section() -> str:
    """
    Build the semantic types section for the LLM extraction prompt.
    Includes all semantic types with UMLS definitions.
    """
    lines = []
    lines.append("=== UMLS SEMANTIC TYPES REFERENCE ===")
    lines.append("Extract entities matching ANY of the following types.\n")

    for group_abbr, group_data in SEMANTIC_GROUPS_WITH_EXAMPLES.items():
        group_name = group_data["group_name"]
        lines.append(f"[{group_abbr}] {group_name}")

        for tui, type_name, definition in group_data["types"]:
            if definition:
                lines.append(f"  - {tui} {type_name}: {definition}")
            else:
                lines.append(f"  - {tui} {type_name}")

        lines.append("")

    return "\n".join(lines)


def load_semantic_groups_from_file(filepath: str = None) -> tuple[dict, dict]:
    """
    Load TUI -> group_abbr and TUI -> type_name mappings from semantic_type_of_UMLS.json.
    """
    semantic_types_path = Path(filepath) if filepath else SEMANTIC_TYPES_JSON
    if not semantic_types_path.is_absolute():
        semantic_types_path = PROJECT_ROOT / semantic_types_path

    tui_to_group = {}
    tui_to_name = {}

    semantic_groups = _load_semantic_types_from_json(semantic_types_path)
    for group_abbr, group_data in semantic_groups.items():
        for tui, type_name, _definition in group_data["types"]:
            tui_to_group[tui] = group_abbr
            tui_to_name[tui] = type_name

    return tui_to_group, tui_to_name


def get_all_tuis() -> set[str]:
    """Return the set of all TUIs defined in the semantic types mapping."""
    tuis = set()
    for group_data in SEMANTIC_GROUPS_WITH_EXAMPLES.values():
        for tui, _type_name, _definition in group_data["types"]:
            tuis.add(tui)
    return tuis
