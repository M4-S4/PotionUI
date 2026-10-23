from typing import Any, List, Tuple

PROMPT_COLUMNS = ("form_data", "prompt_state")
PROMPT_SEPARATOR = "\x1f"


def prompt_texts(data: Any) -> List[str]:
    found: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                lowered = str(key).lower()
                if isinstance(value, str):
                    if "prompt" in lowered and "negative" not in lowered:
                        found.append(value)
                else:
                    walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(data)
    return found


def prompt_column_contains_sql(column: str) -> str:
    return (
        f"CASE WHEN json_valid({column}) THEN EXISTS (SELECT 1 FROM json_tree({column}) jt "
        "WHERE jt.type = 'text' AND LOWER(jt.key) LIKE '%prompt%' AND LOWER(jt.key) NOT LIKE '%negative%' "
        "AND instr(LOWER(jt.value), LOWER(?)) > 0) ELSE 0 END"
    )


def prompt_contains_sql(alias: str = "g") -> str:
    prefix = f"{alias}." if alias else ""
    return " OR ".join(prompt_column_contains_sql(f"{prefix}{column}") for column in PROMPT_COLUMNS)


def prompt_contains_params(needle: str) -> List[str]:
    return [needle] * len(PROMPT_COLUMNS)


def prompt_search_sql(needle: str, alias: str = "g") -> Tuple[str, List[str]]:
    prefix = f"{alias}." if alias else ""
    if PROMPT_SEPARATOR in needle:
        return f"({prompt_contains_sql(alias)})", prompt_contains_params(needle)
    return (
        f"{prefix}id IN (SELECT pt.generation_id FROM generation_prompt_text pt "
        "WHERE instr(LOWER(pt.text), LOWER(?)) > 0)",
        [needle],
    )
