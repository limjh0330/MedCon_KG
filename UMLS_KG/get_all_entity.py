import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

from umls_client import UMLSClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for dotenv_path in (PROJECT_ROOT / ".env", SCRIPT_DIR / ".env"):
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)

DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "UMLS_KG" / "all_entity_of_UMLS.json"
DEFAULT_CUI_FILE = PROJECT_ROOT / "UMLS_KG" / "all_cuis.txt"


def _load_umls_api_key() -> str:
    api_key = os.getenv('"UMLS_API_KEY"')
    if api_key:
        return api_key.strip().strip('"').strip("'")

    for dotenv_path in (PROJECT_ROOT / ".env", SCRIPT_DIR / ".env"):
        if not dotenv_path.exists():
            continue

        values = dotenv_values(dotenv_path)
        api_key = values.get("UMLS_API_KEY") or values.get('"UMLS_API_KEY"')
        if api_key:
            return str(api_key).strip().strip('"').strip("'")

    raise RuntimeError(
        "UMLS_API_KEY was not found. "
        f"Checked: {PROJECT_ROOT / '.env'} and {SCRIPT_DIR / '.env'}"
    )


def _resolve_path(path_value: str | None, default: Path | None = None) -> Path | None:
    if path_value is None:
        return default

    path = Path(path_value)
    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


def _normalize_cui(value: str) -> str | None:
    text = value.strip().upper()
    if not text:
        return None

    if text.startswith("C") and text[1:].isdigit():
        return f"C{int(text[1:]):07d}"

    if text.isdigit():
        return f"C{int(text):07d}"

    return None


def _load_cuis_from_text(path: Path) -> list[str]:
    cuis = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            normalized = _normalize_cui(line)
            if normalized:
                cuis.append(normalized)
    return cuis


def _load_cuis_from_json(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    cuis = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                normalized = _normalize_cui(item)
                if normalized:
                    cuis.append(normalized)
            elif isinstance(item, dict):
                for key in ("CUI", "cui", "ui"):
                    if key in item:
                        normalized = _normalize_cui(str(item[key]))
                        if normalized:
                            cuis.append(normalized)
                        break

    return cuis


def _load_cuis_from_jsonl(path: Path) -> list[str]:
    cuis = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                for key in ("CUI", "cui", "ui"):
                    if key in item:
                        normalized = _normalize_cui(str(item[key]))
                        if normalized:
                            cuis.append(normalized)
                        break
    return cuis


def _load_cuis_from_csv(path: Path) -> list[str]:
    cuis = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return cuis

        candidate_columns = [name for name in reader.fieldnames if name.lower() in {"cui", "ui"}]
        if not candidate_columns:
            return cuis

        target_column = candidate_columns[0]
        for row in reader:
            value = row.get(target_column, "")
            normalized = _normalize_cui(value)
            if normalized:
                cuis.append(normalized)
    return cuis


def _load_cuis_from_file(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".list"}:
        return _load_cuis_from_text(path)
    if suffix == ".json":
        return _load_cuis_from_json(path)
    if suffix == ".jsonl":
        return _load_cuis_from_jsonl(path)
    if suffix == ".csv":
        return _load_cuis_from_csv(path)

    raise ValueError(f"Unsupported CUI file format: {path.suffix}")


def _generate_cuis_from_range(start: int, end: int, width: int) -> list[str]:
    if start > end:
        raise ValueError("start must be less than or equal to end")

    return [f"C{number:0{width}d}" for number in range(start, end + 1)]


def _unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _extract_entity_record(cui: str, concept: dict) -> dict:
    semantic_types = concept.get("semanticTypes", [])
    semantic_type_names = []
    for item in semantic_types:
        name = item.get("name", "").strip()
        if name and name not in semantic_type_names:
            semantic_type_names.append(name)

    relation_count = concept.get("relationCount", 0)
    if isinstance(relation_count, str) and relation_count.isdigit():
        relation_count = int(relation_count)

    return {
        "CUI": cui,
        "semantic_types_name": semantic_type_names,
        "name": concept.get("name", ""),
        "relation_count": relation_count,
    }


def _load_existing_entities(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data if isinstance(data, list) else []


def _save_entities(path: Path, entities: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(entities, file, ensure_ascii=False, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch UMLS concept entities from known CUI values using the UMLS /concept endpoint. "
            "The REST API does not provide a CUI enumeration endpoint, so you must supply a CUI file "
            "or a numeric CUI range."
        )
    )
    parser.add_argument("--cui-file", type=str, default=str(DEFAULT_CUI_FILE))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--start", type=int, help="Start of numeric CUI range, e.g. 1 for C0000001")
    parser.add_argument("--end", type=int, help="End of numeric CUI range")
    parser.add_argument("--width", type=int, default=7, help="Zero-padding width for numeric CUI ranges")
    parser.add_argument("--save-every", type=int, default=100, help="Save output every N successful entities")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file and skip saved CUIs")
    return parser.parse_args()


def main() -> None:
    start = 1
    end = 9999999
    args = _parse_args()
    api_key = _load_umls_api_key()
    umls_system = UMLSClient(api_key)

    output_path = _resolve_path(args.output, DEFAULT_OUTPUT_FILE)
    cui_file_path = _resolve_path(args.cui_file, DEFAULT_CUI_FILE)

    cui_values = []
    if cui_file_path is not None and cui_file_path.exists():
        cui_values.extend(_load_cuis_from_file(cui_file_path))

    
    cui_values.extend(_generate_cuis_from_range(start, end, args.width))

    cui_values = _unique_in_order(cui_values)
    if not cui_values:
        raise RuntimeError(
            "No CUI values were provided. "
            "The UMLS /concept endpoint only retrieves a known CUI, so provide --cui-file or --start/--end."
        )

    entities = []
    processed_cuis = set()
    if args.resume and output_path.exists():
        entities = _load_existing_entities(output_path)
        processed_cuis = {item.get("CUI", "") for item in entities if isinstance(item, dict)}

    missing_count = 0
    for index, cui in enumerate(cui_values, start=1):
        if cui in processed_cuis:
            continue

        concept = umls_system.get_concept(cui)
        if not concept:
            missing_count += 1
            print(f"[{index}/{len(cui_values)}] missing: {cui}")
            continue

        entity = _extract_entity_record(cui, concept)
        entities.append(entity)
        processed_cuis.add(cui)

        if len(entities) % args.save_every == 0:
            _save_entities(output_path, entities)

        print(
            f"[{index}/{len(cui_values)}] saved: {cui} | "
            f"{entity['name']} | semantic_types={entity['semantic_types_name']} | "
            f"relation_count={entity['relation_count']}"
        )

    _save_entities(output_path, entities)
    
    print(f"entity count: {len(entities)}")
    print(f"missing cui count: {missing_count}")
    print(f"umls request count: {umls_system.request_count}")
    print(f"saved file: {output_path}")


if __name__ == "__main__":
    main()
