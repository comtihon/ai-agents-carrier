# `plan` step of lexoffice-einzelprojekt-rechnungen (v2) -- dry-run planning only.
#
# Inputs
#   state["invoice_candidates"]  output of the combined_datasource step (its
#                                `bundle` part): {"at": [...], "ati": [...],
#                                "cp": [...], "classification": {id: {...}},
#                                "project_ids": [...]}
#   records()                    lexoffice articles (step `stream: articles`)
#   state["mode"]                {"dry_run": bool, ...}
#
# Row fields: id, sp (self piloted), adr (address), cid (company id), cname,
# lex (company lexofficeId), hold, inv, del, nick, status.
#
# Output: counts, unmapped_reasons, packages_seen, articles_relevant, invoices,
# special, unmapped, on_hold, airteam_not_individual. Nothing is written
# anywhere: this workflow has no write steps.
import re

bundle = state.get("invoice_candidates") or {}
classification = {str(k): v for k, v in (bundle.get("classification") or {}).items()}

PILOT_RE = {
    "airteam": re.compile(r"airteam\s*-?\s*pilotiert", re.I),
    "kunde": re.compile(r"kunden\s*-?\s*pilotiert", re.I),
}
PACKAGE_RE = re.compile(r"\bpaket\s+(xxl|xl|s|m|l)\b", re.I)


def _price(article):
    price = article.get("price") or {}
    if isinstance(price, dict):
        for key in ("netPrice", "net_price", "netAmount"):
            if price.get(key) is not None:
                return float(price[key])
    try:
        return float(article.get("netPrice") or 0)
    except (TypeError, ValueError):
        return 0.0


# --- lexoffice articles: title/desc/price, archived skipped ----------------
articles = []
for art in records():
    if not isinstance(art, dict) or art.get("archived"):
        continue
    title = art.get("title") or ""
    desc = art.get("description") or art.get("desc") or ""
    text = f"{title}\n{desc}"
    pilot = next((p for p, rx in PILOT_RE.items() if rx.search(text)), None)
    pkg = PACKAGE_RE.search(text)
    if pilot is None or pkg is None:
        continue
    articles.append({
        "id": art.get("id"), "title": title, "desc": desc,
        "price": _price(art), "pilot": pilot, "package": pkg.group(1).lower(),
    })

# --- project rows from the three selections --------------------------------
rows = {}
for bucket in ("at", "ati", "cp"):
    for row in bundle.get(bucket) or []:
        if isinstance(row, dict) and row.get("id") is not None:
            rows.setdefault(str(row["id"]), {**row, "_bucket": bucket})

invoices = {}
special, unmapped, on_hold, airteam_not_individual = [], [], [], []
unmapped_reasons = {}
packages_seen = {}


def _unmapped(row, reason, **extra):
    unmapped_reasons[reason] = unmapped_reasons.get(reason, 0) + 1
    unmapped.append({"id": row.get("id"), "company": row.get("cname"), "reason": reason, **extra})


for pid, row in rows.items():
    cls = classification.get(pid) or {}
    if isinstance(cls.get("result"), dict):          # loop record {"id":..,"result":{..}}
        cls = cls["result"]
    name = str(cls.get("name") or "")
    is_individual = bool(cls.get("isIndividual") or cls.get("is_individual") or row["_bucket"] in ("ati", "cp"))
    if row.get("hold"):
        on_hold.append({"id": row.get("id"), "company": row.get("cname")})
        continue
    if "special" in name.lower():
        special.append({"id": row.get("id"), "company": row.get("cname"), "classification": name})
        continue
    if not row.get("sp") and not is_individual:
        airteam_not_individual.append({"id": row.get("id"), "company": row.get("cname")})
        continue
    pkg = PACKAGE_RE.search(name)
    if pkg is None:
        _unmapped(row, "no_package_in_classification", classification=name)
        continue
    size = pkg.group(1).lower()
    packages_seen[size] = packages_seen.get(size, 0) + 1
    pilot = "kunde" if row.get("sp") else "airteam"
    matches = [a for a in articles if a["pilot"] == pilot and a["package"] == size]
    if not matches:
        _unmapped(row, "no_article_match", package=size, pilot=pilot)
        continue
    if len(matches) > 1:
        _unmapped(row, "ambiguous_article_match", package=size, pilot=pilot,
                  articles=[a["id"] for a in matches])
        continue
    if not row.get("lex"):
        _unmapped(row, "company_without_lexoffice_id")
        continue
    art = matches[0]
    line_desc = art["desc"].replace("XXX", f"{row.get('adr') or ''}, Projekt-ID {row.get('id')}")
    price = art["price"] or float(cls.get("price") or 0)
    inv = invoices.setdefault(str(row.get("cid")), {
        "company_id": row.get("cid"), "company": row.get("cname"),
        "lexoffice_id": row.get("lex"), "lines": [], "net_total": 0.0,
    })
    inv["lines"].append({
        "project_id": row.get("id"), "article_id": art["id"], "title": art["title"],
        "description": line_desc, "net_price": price, "package": size, "pilot": pilot,
    })
    inv["net_total"] = round(inv["net_total"] + price, 2)

invoice_list = sorted(invoices.values(), key=lambda i: str(i["company"] or ""))
output = {
    "dry_run": bool((state.get("mode") or {}).get("dry_run", True)),
    "counts": {
        "projects": len(rows),
        "invoices": len(invoice_list),
        "invoice_lines": sum(len(i["lines"]) for i in invoice_list),
        "special": len(special),
        "unmapped": len(unmapped),
        "on_hold": len(on_hold),
        "airteam_not_individual": len(airteam_not_individual),
        "net_total": round(sum(i["net_total"] for i in invoice_list), 2),
    },
    "unmapped_reasons": unmapped_reasons,
    "packages_seen": packages_seen,
    "articles_relevant": [{k: a[k] for k in ("id", "title", "pilot", "package", "price")} for a in articles],
    "invoices": invoice_list,
    "special": special,
    "unmapped": unmapped,
    "on_hold": on_hold,
    "airteam_not_individual": airteam_not_individual,
}
