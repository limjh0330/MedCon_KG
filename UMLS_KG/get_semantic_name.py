import os
import sys
import json
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


API_KEY = _load_umls_api_key()
umls_system = UMLSClient(API_KEY)

OUTPUT_FILE = PROJECT_ROOT / "UMLS_KG" / "semantic_type_of_UMLS.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

semantic_type_list = []
semantic_rel_type = []
no_tui_list = []
for number in range(1,1000):
    if len(str(number)) < 3:
        number = str(number).zfill(3)

    semantic_type_info = umls_system.umls_semantic_types(number=number)
    if not semantic_type_info:
        no_tui_list.append(f"T{number}")
        print(f"There is no T{number}")
        continue
    else:
        semantic_type_group = semantic_type_info.get("semanticTypeGroup", {})
        semantic_type_group_name = semantic_type_group.get("expandedForm", "")
        semantic_type_group_code = semantic_type_group.get("abbreviation", "")
        semantic_type_class = semantic_type_group.get("classType", "")

        semantic_type_def = semantic_type_info.get("definition", "")        
        semantic_type_name = semantic_type_info.get("name", "")
        semantic_type_examples = semantic_type_info.get("example", [])
        semantic_type = {
            "abbreviation": semantic_type_group_code,
            "semanticTypeGroup": semantic_type_group_name,
            "TUI": f"T{number}",
            "name": semantic_type_name,
            "definition": semantic_type_def,
            "example": semantic_type_examples
        }
        if semantic_type_class == "SemanticGroup":
            if semantic_type in semantic_type_list:
                continue
            else:
                semantic_type_list.append(semantic_type)
        else:
            if semantic_type in semantic_rel_type:
                continue
            else:
                semantic_rel_type.append(semantic_type)



semantic_type_of_umls = [{"semantic_type": semantic_type_list}] + [{"sematic_relation": semantic_rel_type}] + [{"no_tui": no_tui_list}]

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(semantic_type_of_umls, file, ensure_ascii=False, indent=2)

print("semantic type count:", len(semantic_type_list))
print("semantic relation type count:", len(semantic_rel_type))
print("no tui count:", len(no_tui_list))
print(f"saved file: {OUTPUT_FILE}")
