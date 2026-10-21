def field_by_name(fields, name):
    for f in fields:
        if f.get("name") == name:
            return f
        found = field_by_name(f.get("children") or [], name)
        if found is not None:
            return found
    return None


def all_field_names(node):
    found = set()
    for child in node.get("children") or []:
        if child.get("name"):
            found.add(child["name"])
        found |= all_field_names(child)
    return found
