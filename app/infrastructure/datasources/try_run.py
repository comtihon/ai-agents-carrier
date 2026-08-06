"""Single-operation dry run for the datasource editor.

Executes one operation of a (possibly unsaved) definition and returns a
size-capped sample of the raw response together with a meta-LLM-suggested
JMESPath mapping. Target-server and LLM failures are encoded in the result
dict, never raised — mirroring the probe endpoint's contract.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import jmespath

from app.core.config import Settings
from app.domain.models.data_source_definition import DataSourceDefinition
from app.infrastructure.datasources.executor import DataSourceExecutor

logger = logging.getLogger(__name__)

# Response samples are shrunk before being returned / fed to the LLM: lists
# keep their first few elements, long strings are truncated.
_LIST_SAMPLE = 3
_STRING_CAP = 500

_MAPPING_PROMPT = (
    "You are helping configure a REST/GraphQL data source. Below is a sample "
    "of the raw JSON response of one API operation (arrays and long strings "
    "are truncated).\n\n"
    "Sample response:\n{sample}\n\n"
    "Compose a single JMESPath expression that extracts the main list of "
    "items (or the main payload object) and maps each item to its most "
    "useful fields, for example: content[].{{id: id, number: voucherNumber}}. "
    "Reply with ONLY the JMESPath expression on one line — no explanation, "
    "no code fences."
)


def shrink_sample(value: Any, list_limit: int = _LIST_SAMPLE, string_cap: int = _STRING_CAP) -> Any:
    """Recursively cap list lengths and string sizes of a response payload."""
    if isinstance(value, dict):
        return {k: shrink_sample(v, list_limit, string_cap) for k, v in value.items()}
    if isinstance(value, list):
        return [shrink_sample(v, list_limit, string_cap) for v in value[:list_limit]]
    if isinstance(value, str) and len(value) > string_cap:
        return value[:string_cap] + "…"
    return value


async def suggest_mapping(sample: Any, settings: Settings) -> str | None:
    """Ask the meta-LLM for a JMESPath mapping; None when it fails validation."""
    try:
        from langchain_core.messages import HumanMessage

        from app.core.container import build_llm_native

        provider = settings.meta_llm_provider or settings.llm_provider
        llm = build_llm_native(provider, settings.meta_llm_model, settings, max_tokens=300)
        prompt = _MAPPING_PROMPT.format(sample=json.dumps(sample, ensure_ascii=False, default=str))
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content if isinstance(response.content, str) else str(response.content)
        expr = text.strip().strip("`").strip()
        # Some models prefix a label despite instructions.
        if expr.lower().startswith("jmespath"):
            expr = expr[len("jmespath"):].lstrip(": \n")
        expr = expr.splitlines()[0].strip()
        if not expr:
            return None
        jmespath.compile(expr)
        return expr
    except Exception:
        logger.warning("meta-LLM mapping suggestion failed", exc_info=True)
        return None


async def try_operation(
    definition: DataSourceDefinition,
    operation: str,
    params: dict[str, Any],
    settings: Settings,
    executor: DataSourceExecutor | None = None,
) -> dict[str, Any]:
    """Run *operation* once and return a sampled output + suggested mapping.

    The target operation is executed without its ``mapping``,
    ``response_schema`` and ``paginate`` blocks so the caller sees one raw
    page of the API's real response. Upstream operations referenced through
    templates run unchanged.
    """
    target = definition.get_operation(operation)
    if target is None:
        raise ValueError(f"Data source '{definition.id}' has no operation '{operation}'")

    stripped = target.model_copy(update={"mapping": None, "response_schema": None, "paginate": None})
    trial = definition.model_copy(
        update={
            "operations": [stripped if op.name == operation else op for op in definition.operations],
            # A dry run must never serve or seed the shared cache.
            "cache": type(definition.cache)(ttl_seconds=0),
        }
    )

    try:
        raw = await (executor or DataSourceExecutor()).execute(trial, operation, params)
    except Exception as exc:
        logger.info("try-operation '%s' failed: %s", operation, exc)
        return {"status": "error", "error": str(exc), "api_output": None, "suggested_mapping": None}

    sample = shrink_sample(raw)
    mapping = await suggest_mapping(sample, settings)
    return {"status": "ok", "error": None, "api_output": sample, "suggested_mapping": mapping}
