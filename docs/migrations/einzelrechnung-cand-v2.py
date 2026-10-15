# combined_datasource part `cand` of lexoffice-einzelprojekt-rechnungen (v2).
#
# Union of the project ids selected by `sel_ati` (airteam-piloted, individual)
# and `sel_cp` (customer-piloted, individual), excluding projects on hold.
# Replaces the 80 copy-pasted classification "slot" steps: the next part,
# `get_classification`, loops over `project_ids` itself, so there is no cap.
#
# In scope: `parts` (outputs of earlier parts of the combined step).


def _rows(value):
    """The selected rows, whatever envelope the sel script wraps them in."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("rows", "projects", "selected", "items"):
            if isinstance(value.get(key), list):
                return value[key]
    return []


seen = set()
project_ids = []
on_hold = []
for part_id in ("sel_ati", "sel_cp"):
    for row in _rows(parts.get(part_id)):
        pid = row.get("id") if isinstance(row, dict) else row
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        if isinstance(row, dict) and row.get("hold"):
            on_hold.append(pid)
            continue
        project_ids.append(pid)

output = {"project_ids": project_ids, "excluded_on_hold": on_hold}
