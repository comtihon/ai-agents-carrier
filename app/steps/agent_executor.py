"""Step executor for ``langgraph-agent`` and ``claude-agent`` step types.

Responsibilities
----------------
1. Load the ``AgentDefinition`` from the backend by ``agent_id``.
2. Determine the effective runtime (step ``runtime_override`` > agent ``default_runtime``).
3. Build the input dict from the workflow state using ``input_mapping``.
4. Branch on runtime:
   - **local** — run inline via ``app.agents.local_agent.run_local_agent``.
     No HTTP, no subprocess.  Returns the output dict directly.
   - **docker / k8s** — spawn the agent HTTP server, send ``POST /start``,
     then suspend via LangGraph ``interrupt()`` until the agent callbacks with
     its output.

HTTP protocol (docker / k8s)
-----------------------------
The backend spawns the agent (``runtime.spawn``) which starts a FastAPI server
on a known port.  The backend then calls::

    POST {agent_url}/start
    {
        "run_id":        "<run_id>",
        "input":         { ... },
        "callback_url":  "<backend_base_url>",
        "agent_config":  {
            "system_prompt": "...",
            "model":         "...",
            "tools":         [...],
            "mcp_servers":   [...],
            "credentials":   {...},
            "extra":         {...}
        }
    }

The agent runs asynchronously and, when done, calls back to the backend::

    POST {callback_url}/api/v1/runs/{run_id}/agent/output
    { "output": { ... } }

That callback endpoint resumes the paused LangGraph run via
``Command(resume={"output": raw_output})``.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import httpx


class MetaLLMRejectionError(RuntimeError):
    """Raised when meta-LLM quality gate rejects a step's output.

    Carries the already-mapped workflow state so callers can surface the
    agent's actual output alongside the rejection reason instead of discarding it.
    """
    def __init__(self, message: str, mapped_result: dict[str, Any], reason: str) -> None:
        super().__init__(message)
        self.mapped_result = mapped_result
        self.reason = reason
from langgraph.types import interrupt

from app.core.config import get_settings

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.domain.models.agent_definition import AgentDefinition
    from app.infrastructure.persistence.agent_backend import AgentDefinitionBackend
    from app.runtime.base import AgentRuntime

logger = logging.getLogger(__name__)

# --- Tools addon: bash-level credential/binary gating ---
# The "tools" addon toggles which bash-level integrations an agent may use.
# Backend ALWAYS sends tool_access={"github":b,"jira":b,"graphify":b} (all
# false when the addon is absent). Runtimes treat absent/null tool_access as
# ALL ENABLED (rollout compat) and a present dict as exact (missing key =
# disabled). When a tool is disabled backend-side we pop its credential keys
# out of the resolved credentials dict so the agent never receives them.
_KNOWN_TOOLS: tuple[str, ...] = ("github", "jira", "graphify")
_TOOL_CREDENTIAL_KEYS: dict[str, set[str]] = {
    "github": {"MCP_GITHUB_API_KEY", "GITHUB_TOKEN"},
    "jira": {"MCP_JIRA_API_TOKEN", "JIRA_API_TOKEN", "JIRA_URL", "JIRA_USERNAME"},
}

_COMPRESSION_INSTRUCTIONS: dict[str, str] = {
    "lite": (
        "Be concise. Drop filler words (just/really/basically/actually/simply). "
        "Fragments OK. Skip pleasantries and hedging."
    ),
    "full": (
        "Respond terse like a smart caveman. Drop: articles (a/an/the), filler words, "
        "pleasantries, hedging. Fragments OK. Use short synonyms (big not extensive, "
        "fix not 'implement a solution for'). Technical terms exact. Code blocks unchanged. "
        "Pattern: [thing] [action] [reason]. [next step]."
    ),
    "ultra": (
        "Maximum compression. Single words / symbols where possible. "
        "No articles, no filler, no fluff. Abbreviate freely. Code blocks exact and unchanged."
    ),
}

_PREVIOUS_TASK_GUIDANCE = (
    "You previously ran with this input and produced this output. Analyze the new "
    "input against your previous output (e.g. answers received vs questions you "
    "asked). If your previous open points are resolved, continue the original job "
    "from where you left off using the previous output plus the new information. "
    "If key questions remain unanswered or unclear, emit context_sufficient=false "
    "with the unanswered questions."
)


def _truncate_for_prompt(value: Any, cap: int = 4000) -> Any:
    """Walk ``value`` and truncate long string fields so they stay prompt-sized.

    Handles bare strings and top-level dict values (one level of nesting is
    enough for the ``input``/``final output`` blobs this is used for). Leaves
    non-str values (numbers, bools, nested dicts/lists) untouched.
    """
    def _cap(s: str) -> str:
        if len(s) <= cap:
            return s
        return s[:cap] + "…[truncated]"

    if isinstance(value, str):
        return _cap(value)
    if isinstance(value, dict):
        return {k: (_cap(v) if isinstance(v, str) else v) for k, v in value.items()}
    return value


async def _build_previous_task(
    repo: Any, run_id: str, step_id: str, visit_count: int
) -> dict[str, Any] | None:
    """Build the ``previous_task`` context block from the prior visit's task doc.

    NOTE: this looks up the prior visit using the plain (no clarification
    discriminator) key ``f"{run_id}_{step_id}_{visit_count - 1}"``. If the
    prior visit itself carried a clarification discriminator (i.e. it was
    re-run after answers), this plain-key lookup will miss and we return
    ``None`` — acceptable no-op for v1, documented here rather than solved.
    """
    if visit_count <= 0:
        return None
    prior = await repo.get_task(f"{run_id}_{step_id}_{visit_count - 1}")
    if not prior or prior.get("status") != "finished":
        return None
    final = None
    for o in reversed(prior.get("outputs", []) or []):
        if isinstance(o, dict) and o.get("type") == "final":
            final = o.get("content")
            break
    if final is None:
        return None
    return {
        "guidance": _PREVIOUS_TASK_GUIDANCE,
        "input": _truncate_for_prompt(prior.get("input")),
        "output": _truncate_for_prompt(final),
    }


def _build_agent_config(
    agent_def: "AgentDefinition",
    settings: "Settings",
    step: dict[str, Any] | None = None,
    run_id: str | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``agent_config`` payload to forward in ``POST /start``.

    Used only for docker / k8s agents.

    The payload is built from ``agent_def.agent_input`` (which is passed
    through wholesale as ``extra`` and also promoted to top-level fields when
    recognised keys are present), plus ``mcp_servers`` and ``credentials``
    resolved from ``settings``.

    Security note: credential values are resolved strings (not env-var
    references) because the agent runs in a trusted container in the same
    cluster.  For production Kubernetes deployments, mounting K8s Secrets as
    env vars is a more secure alternative.

    Resolution rules
    ----------------
    - ``system_prompt``, ``model``, ``tools``: promoted from ``agent_input``
      when present; otherwise ``None``.
    - ``mcp_servers``: built from ``settings.get_mcp_integrations()``.
      Filtered to ``agent_input["tools"]`` when that key is provided.
    - ``credentials``: API keys from every active LLM integration.
    - ``tool_access``: ``{tool: bool}`` for every known bash-level tool
      (github/jira/graphify). Derived from the agent's ``tools`` addon —
      no addon means strict: all disabled. Credential keys for disabled
      tools are popped from ``credentials`` before it is returned.
    - ``extra``: the entire ``agent_input`` dict forwarded as-is.
    - ``description`` is NOT included.

    Note: step-level ``env_vars`` with ``from_config`` are NOT gated here —
    that path is an explicit operator override and deliberately bypasses the
    tools-addon credential gating above.
    """
    from app.core.config import McpIntegrationConfig

    agent_input: dict[str, Any] = agent_def.agent_input or {}

    # Promote known keys from agent_input.
    system_prompt = agent_input.get("system_prompt")
    model = agent_input.get("model")
    tools = agent_input.get("tools")
    llm_provider_name = agent_input.get("llm_provider") or settings.llm_provider

    # Resolve provider → inject base_url + api_key_env into extra so the agent
    # can route to the right LLM without any hardcoded heuristics.
    extra: dict[str, Any] = dict(agent_input)
    if llm_provider_name:
        intg = settings.get_llm_integration(llm_provider_name)
        if intg:
            extra["llm_base_url"] = intg.base_url
            extra["llm_api_key_env"] = intg.resolved_api_key_env()

    # Inject S3 workspace config when the agent has an s3 addon attached.
    if agent_def.s3_addon is not None:
        s3_addon = agent_def.s3_addon
        s3_path = s3_addon.path.replace("{workflow_id}", run_id or "")
        s3_path = s3_path.replace("{project_id}", (state or {}).get("project_id", ""))
        extra["s3_bucket"] = s3_addon.bucket
        extra["s3_path"] = s3_path

    # Apply compression level: step config takes priority, then agent_input, then none.
    compression_level = (step or {}).get("compression_level") or agent_input.get("compression_level", "none")
    compression_instruction = _COMPRESSION_INSTRUCTIONS.get(compression_level or "none", "")
    if compression_instruction:
        system_prompt = (
            f"{compression_instruction}\n\n{system_prompt}"
            if system_prompt
            else compression_instruction
        )

    # Inject Output Protocol when the step declares an output_mapping.
    output_mapping = (step or {}).get("output_mapping") or {}
    _protocol_keys = list(output_mapping)
    if _protocol_keys:
        field_list = "\n".join(f"- {k}" for k in _protocol_keys)
        protocol = (
            "\n\n## Output Protocol\n\n"
            "Your output MUST include these fields (YAML or JSON — the orchestrator "
            "extracts either format):\n\n"
            f"{field_list}"
        )
        system_prompt = f"{system_prompt}{protocol}" if system_prompt else protocol.lstrip()

    # --- MCP servers ---
    raw_integrations: list[McpIntegrationConfig] = settings.get_mcp_integrations()
    if agent_def.mcp_addon is None:
        enabled_mcp: set[str] | None = set()  # no addon → no MCPs
    else:
        enabled_mcp = agent_def.mcp_addon.enabled_servers()
    mcp_servers: list[dict[str, Any]] = []
    for intg in raw_integrations:
        if intg.name not in enabled_mcp:
            continue
        entry: dict[str, Any] = {"name": intg.name, "transport": intg.transport, "env": intg.env}
        if intg.transport == "stdio":
            cmd = [intg.command] if intg.command else []
            cmd += intg.args
            entry["command"] = cmd
        else:
            entry["url"] = intg.url
            if intg.api_key:
                entry["api_key"] = intg.api_key
        mcp_servers.append(entry)

    # --- Credentials (resolved API-key values) ---
    credentials: dict[str, str] = {}
    for llm_intg in settings.get_llm_integrations():
        key = llm_intg.resolved_api_key()
        if key:
            credentials[llm_intg.resolved_api_key_env()] = key
    # Also forward standalone API keys set directly (not via LLM_INTEGRATIONS)
    for key_name, val in settings.get_forwardable_config().items():
        if key_name not in credentials:
            credentials[key_name] = val

    # --- Tools addon gating ---
    # Build tool_access for every known tool. No addon → strict: all disabled.
    # Then pop credential keys for disabled tools so the agent never sees them.
    # NEVER touch LLM / non-tool credentials (ANTHROPIC / OPENROUTER /
    # GOOGLE_APPLICATION_CREDENTIALS_JSON / HUBSPOT_TOKEN / HF_TOKEN etc.) —
    # only the keys enumerated in _TOOL_CREDENTIAL_KEYS are ever removed.
    tools_addon = agent_def.tools_addon
    enabled_tools: set[str] = tools_addon.enabled_tools() if tools_addon is not None else set()
    tool_access: dict[str, bool] = {name: (name in enabled_tools) for name in _KNOWN_TOOLS}
    for tool_name, cred_keys in _TOOL_CREDENTIAL_KEYS.items():
        if not tool_access.get(tool_name, False):
            for cred_key in cred_keys:
                credentials.pop(cred_key, None)

    # An ENABLED tool needs its full bash-level credential set, not just the
    # token: curl against $JIRA_URL requires the endpoint and identity vars,
    # which get_forwardable_config() never picks up (no credential suffix).
    if tool_access.get("jira"):
        for cred_key, val in (
            ("JIRA_URL", settings.mcp_jira_jira_url),
            ("JIRA_USERNAME", settings.mcp_jira_username),
            ("JIRA_API_TOKEN", settings.mcp_jira_api_token),
        ):
            if val and not credentials.get(cred_key):
                credentials[cred_key] = val
    if tool_access.get("github"):
        if settings.mcp_github_api_key and not credentials.get("GITHUB_TOKEN"):
            credentials["GITHUB_TOKEN"] = settings.mcp_github_api_key

    # Resolve env_vars from step config
    env_vars: dict[str, str] = {}
    if step:
        forwardable = settings.get_forwardable_config()
        for entry in (step.get("env_vars") or []):
            name = entry.get("name", "").strip()
            if not name:
                continue
            if "from_config" in entry:
                val = forwardable.get(entry["from_config"])
                if val:
                    env_vars[name] = val
            elif "value" in entry:
                env_vars[name] = str(entry["value"])

    # Forward meta-LLM settings to agent os.environ
    if hasattr(settings, 'meta_llm_provider') and settings.meta_llm_provider:
        env_vars["META_LLM_PROVIDER"] = settings.meta_llm_provider
    if hasattr(settings, 'meta_llm_model') and settings.meta_llm_model:
        env_vars["META_LLM_MODEL"] = settings.meta_llm_model

    return {
        "system_prompt": system_prompt,
        "model": model,
        "tools": tools,
        "mcp_servers": mcp_servers,
        "credentials": credentials,
        "tool_access": tool_access,
        "extra": extra,
        "env_vars": env_vars,
        "expected_output_fields": list(_protocol_keys),
    }


def _apply_usage_keys(
    target: dict[str, Any],
    step_id: str,
    raw_output: Any,
    judge_usage: dict[str, int] | None,
) -> None:
    """Write the three token-usage buckets into *target* (in place): the
    agent-LLM's own usage, the agent's post-compact meta-LLM usage (both keyed
    per step), and the backend judge (meta-LLM evaluator) usage (a single
    workflow-wide bucket). The buckets are never merged into one another; a
    key is omitted entirely when its source has no data — state-level
    ``_sum_usage`` reducers accumulate across workflow-loop re-executions.
    """
    if isinstance(raw_output, dict):
        agent_usage = raw_output.get("token_usage")
        if agent_usage:
            target[f"_agent_token_usage_{step_id}"] = agent_usage
        meta_usage = raw_output.get("meta_token_usage")
        if meta_usage:
            target[f"_meta_token_usage_{step_id}"] = meta_usage
    if judge_usage:
        target["_judge_token_usage"] = judge_usage


def _apply_mapping(
    source: dict[str, Any],
    mapping: dict[str, str] | None,
) -> dict[str, Any]:
    """Apply a key-mapping dict to *source*, returning a new dict.

    When *mapping* is ``None`` or empty the entire *source* is returned as-is.
    Otherwise only the keys listed in *mapping* are included, renamed according
    to the mapping::

        # mapping: {"workflow_key": "agent_key"}
        source = {"plan": "...", "request": "..."}
        mapping = {"plan": "task_description"}
        # result = {"task_description": "..."}
    """
    if not mapping:
        return dict(source)
    return {
        agent_key: source[workflow_key]
        for workflow_key, agent_key in mapping.items()
        if workflow_key in source
    }


async def _meta_llm_evaluate(
    raw_output: Any,
    input_data: Any,
    step_id: str,
    settings: "Settings",
    success_criteria: str | None = None,
    fail_criteria: str | None = None,
) -> dict:
    """Evaluate step output against optional criteria using a lightweight LLM.

    Returns: {"passed": bool, "reason": str, "usage": dict | None}
    ``usage``, when captured, has the shape
    ``{"input_tokens": int, "output_tokens": int, "total_tokens": int}`` — it is
    ``None`` when the underlying LLM client doesn't expose usage metadata or the
    call errored (no real call succeeded).
    Always returns passed=True on any failure (non-blocking).
    """
    try:
        from app.core.container import build_llm_native
        from langchain_core.messages import HumanMessage
        provider = settings.meta_llm_provider or settings.llm_provider
        model = settings.meta_llm_model
        llm = build_llm_native(provider, model, settings, max_tokens=512)

        request_text = (
            input_data.get("request")
            or input_data.get("task")
            or input_data.get("prompt")
            or str(input_data)
        ) if isinstance(input_data, dict) else str(input_data)
        output_text = (
            raw_output.get("result") or raw_output.get("answer") or str(raw_output)
            if isinstance(raw_output, dict) else str(raw_output)
        )

        has_criteria = bool(success_criteria or fail_criteria)
        if has_criteria:
            criteria_lines = []
            if success_criteria:
                criteria_lines.append(f"Success criteria: {success_criteria}")
            if fail_criteria:
                criteria_lines.append(f"Fail criteria: {fail_criteria}")
            criteria_text = "\n".join(criteria_lines)
            prompt = (
                "You are evaluating whether an AI agent step produced acceptable output.\n\n"
                f"Original request: {request_text}\n\n"
                f"Agent output:\n{output_text}\n\n"
                f"{criteria_text}\n\n"
                "Based on the criteria above, did the agent output PASS or FAIL?\n"
                "Respond with ONLY:\n"
                "DECISION: PASS or DECISION: FAIL\n"
                "REASON: <one line>"
            )
        else:
            prompt = (
                "You are evaluating whether an AI agent step produced useful output "
                "as part of a multi-step automated workflow.\n\n"
                f"Task context for this step: {request_text}\n\n"
                f"Agent output:\n{output_text}\n\n"
                "IMPORTANT: this agent is ONE step in a larger pipeline. "
                "It is NOT expected to complete the entire end-to-end task on its own. "
                "A researcher step should produce research/analysis. "
                "A planner step should produce a plan. "
                "A coder step should produce code changes. Etc.\n\n"
                "PASS if the output contains real, substantive content relevant to this step's role "
                "(e.g. a researcher step must have actual ticket data, repo URLs, file references — "
                "not placeholders like 'pending', 'unknown', 'cannot access').\n"
                "FAIL if the output is empty, an error/blocker message, fabricated placeholders, "
                "or lacks the actual data the step was supposed to gather.\n"
                "A researcher that says 'no Jira access' or 'pending ticket data' is a FAIL — "
                "it must have real ticket context and real repo info to pass.\n"
                "Do NOT fail because the agent did not implement the full end-to-end solution.\n"
                "Respond with ONLY:\n"
                "DECISION: PASS or DECISION: FAIL\n"
                "REASON: <one line>"
            )

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content if isinstance(response.content, str) else str(response.content)

        # Capture token usage off the AIMessage — modern LangChain chat models
        # (ChatAnthropic/ChatOpenAI/ChatGoogleGenerativeAI, all used by
        # build_llm_native) populate `usage_metadata` with
        # input_tokens/output_tokens/total_tokens. None when unavailable.
        usage: dict[str, int] | None = None
        _usage_meta = getattr(response, "usage_metadata", None)
        if _usage_meta:
            usage = {
                "input_tokens": _usage_meta.get("input_tokens", 0),
                "output_tokens": _usage_meta.get("output_tokens", 0),
                "total_tokens": _usage_meta.get("total_tokens", 0),
            }

        passed = True
        reason = ""
        for line in text.strip().splitlines():
            line = line.strip()
            if line.startswith("DECISION:"):
                raw_d = line[len("DECISION:"):].strip().upper()
                passed = (raw_d != "FAIL")
            elif line.startswith("REASON:"):
                reason = line[len("REASON:"):].strip()

        logger.info("[step '%s'] meta-LLM evaluation: %s — %s", step_id, "PASS" if passed else "FAIL", reason)
        return {"passed": passed, "reason": reason, "usage": usage}

    except Exception as exc:
        logger.warning("[step '%s'] meta-LLM evaluation failed: %s — defaulting to pass", step_id, exc)
        return {"passed": True, "reason": "evaluator error, defaulting to pass", "usage": None}


async def execute_agent_step(
    step: dict[str, Any],
    state: dict[str, Any],
    agent_backend: "AgentDefinitionBackend",
    run_id: str,
    callback_base_url: str,
    settings: "Settings | None" = None,
    progress_cb: Callable[[str], Awaitable[None]] | None = None,
    run_repository: Any = None,
    pvc_lease_repository: Any = None,
    agent_task_repository: Any = None,
    warm_pod_repository: Any = None,
    use_meta_llm: bool = True,
) -> dict[str, Any]:
    """Execute a ``langgraph-agent`` or ``claude-agent`` step.

    Parameters
    ----------
    step:
        The raw step dict from the workflow YAML.  Required keys:

        - ``id``       — step identifier (for logging)
        - ``agent_id`` — references an ``AgentDefinition`` in the backend

        Optional keys:

        - ``runtime_override``  — overrides ``agent_def.default_runtime``
        - ``image_override``    — overrides ``agent_def.image`` (docker only)
        - ``input_mapping``     — ``{workflow_key: agent_key}`` dict
        - ``output_mapping``    — ``{agent_key: workflow_key}`` dict
        - ``output_key``        — single key to store the whole output dict
          (used when ``output_mapping`` is absent)
    state:
        Current workflow state dict.
    agent_backend:
        The ``AgentDefinitionBackend`` to look up the ``AgentDefinition``.
    run_id:
        The workflow run ID — passed to the agent and used for the callback URL.
    callback_base_url:
        Base URL of the backend (e.g. ``http://localhost:8000``).  The agent
        uses this to call back with its output.
    settings:
        App ``Settings`` instance.  Resolved lazily if not provided.
    progress_cb:
        Optional async callback for progress messages (local runtime only).

    Returns
    -------
    dict
        A partial state update dict to be merged into the workflow state.

    Raises
    ------
    ValueError
        When the agent definition is not found, the runtime type is unknown,
        or the agent output cannot be parsed.
    """
    step_id: str = step["id"]
    agent_id: str = step["agent_id"]

    _visit_count = int((state.get("_visit_counts") or {}).get(step_id, 0))
    task_key = f"{run_id}_{step_id}_{_visit_count}"
    if state.get("_clarification_answers"):
        _ans_hash = hashlib.sha1(json.dumps(state["_clarification_answers"], sort_keys=True, default=str).encode()).hexdigest()[:8]
        task_key += f"_c{_ans_hash}"

    # --- 1. Load agent definition ---
    agent_def: AgentDefinition | None = await agent_backend.get(agent_id)
    if agent_def is None:
        raise ValueError(
            f"[step '{step_id}'] AgentDefinition '{agent_id}' not found. "
            "Register the agent via POST /api/v1/agents before using it in a workflow."
        )

    # --- 2. Determine effective runtime ---
    runtime_type: str = step.get("runtime_override") or agent_def.default_runtime
    logger.info(
        "[step '%s'] agent='%s' runtime='%s' run_id='%s'",
        step_id, agent_id, runtime_type, run_id,
    )

    # --- 3. Resolve settings ---
    if settings is None:
        settings = get_settings()

    # --- 4. Build input from state via input_mapping ---
    input_mapping: dict[str, str] | None = step.get("input_mapping")
    input_data: dict[str, Any] = _apply_mapping(state, input_mapping)
    # If a previous ask_context interrupt was answered, fold those answers into
    # the input so the agent can use them as clarifying context.
    if state.get("_clarification_answers"):
        input_data = {**input_data, "clarification_context": state["_clarification_answers"]}
    if _visit_count > 0 and agent_task_repository is not None:
        _prev_task = await _build_previous_task(agent_task_repository, run_id, step_id, _visit_count)
        if _prev_task:
            input_data = {**input_data, "previous_task": _prev_task}

    # --- 5. Branch on runtime ---
    if runtime_type == "local":
        # Run inline — no HTTP, no subprocess.
        from app.agents.local_agent import run_local_agent

        logger.info("[step '%s'] running local inline agent", step_id)
        allowed_mcp = agent_def.mcp_addon.enabled_servers() if agent_def.mcp_addon else set()
        raw_output = await run_local_agent(
            agent_input=agent_def.agent_input,
            input_data=input_data,
            settings=settings,
            progress_cb=progress_cb,
            compression_level=step.get("compression_level", "none"),
            allowed_mcp=allowed_mcp,
        )
        logger.info("[step '%s'] local agent completed, output keys: %s", step_id, list(raw_output))
    else:
        # Docker / K8s: use the HTTP protocol with interrupt-based suspension.
        from app.runtime.factory import get_runtime

        runtime: AgentRuntime = get_runtime(
            runtime_type,
            registry_username=settings.docker_registry_username,
            registry_password=settings.docker_registry_password,
            agent_namespace=settings.agent_namespace,
            callback_override_url=settings.agent_callback_url,
        )
        agent_config_payload = _build_agent_config(agent_def, settings, step=step, run_id=run_id, state=state)
        resolved_env_vars: dict[str, str] = agent_config_payload.get("env_vars") or {}

        # --- Warm pod reuse: check if a warm pod is available for this agent ---
        _is_warm_reuse = False
        _warm_agent_url: str | None = None
        if warm_pod_repository is not None and runtime_type == "k8s":
            _warm_record = await warm_pod_repository.get(run_id, agent_id)
            if _warm_record is not None:
                try:
                    async with httpx.AsyncClient() as _hc:
                        _hr = await _hc.get(f"{_warm_record.agent_url}/health", timeout=5.0)
                        if _hr.status_code == 200:
                            _is_warm_reuse = True
                            _warm_agent_url = _warm_record.agent_url
                        else:
                            # Unhealthy pod — delete stale record, fall through to spawn
                            try:
                                await warm_pod_repository.delete(run_id, agent_id)
                            except Exception:
                                pass
                except Exception:
                    # Pod gone — delete the stale record for this specific agent
                    try:
                        await warm_pod_repository.delete(run_id, agent_id)
                    except Exception:
                        pass

        # LangGraph reruns the node from scratch on resume. Detect this by checking
        # whether a container already exists for this run (spawned in the first execution).
        # If so, skip spawn+start — interrupt() will return immediately with the stored output.
        _is_resume = (
            not _is_warm_reuse
            and hasattr(runtime, "has_container_for_run")
            and await runtime.has_container_for_run(agent_def, run_id)
        )

        container_callback_url = callback_base_url  # default; overridden below for docker/non-resume
        if _is_warm_reuse:
            # Warm pod still alive — reuse it: send a fresh /start and enter poll loop.
            agent_url = _warm_agent_url  # type: ignore[assignment]
            container_callback_url = runtime.rewrite_callback_url(callback_base_url)
            logger.info("[step '%s'] reusing warm pod at %s for run '%s'", step_id, agent_url, run_id)
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{agent_url}/start",
                        json={
                            "run_id": run_id,
                            "task_id": task_key,
                            "input": input_data,
                            "callback_url": container_callback_url,
                            "agent_config": agent_config_payload,
                        },
                        timeout=10.0,
                    )
                    resp.raise_for_status()
            except Exception as exc:
                raise RuntimeError(
                    f"[step '{step_id}'] Failed to start warm agent at {agent_url}: {exc}"
                ) from exc
            if agent_task_repository is not None:
                from datetime import datetime, timezone
                _task_doc = {
                    "_id": task_key,
                    "run_id": run_id,
                    "step_id": step_id,
                    "agent_id": agent_id,
                    "agent_url": agent_url,
                    "input": input_data,
                    "agent_config": agent_config_payload if isinstance(agent_config_payload, dict) else {},
                    "status": "pending",
                    "loop_count": 0,
                    "max_loops": get_settings().agent_max_loops,
                    "outputs": [],
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
                try:
                    await agent_task_repository.save_task(_task_doc)
                except Exception as _e:
                    logger.warning("[step '%s'] failed to save agent task (warm reuse): %s", step_id, _e)
        elif not _is_resume:
            # --- 5a. Spawn the agent HTTP server ---
            agent_url = await runtime.spawn(agent_def, step, run_id, callback_base_url, extra_env=resolved_env_vars)
            logger.info("[step '%s'] agent server spawned at %s", step_id, agent_url)

            # For Docker, the agent runs in a container where localhost = itself.
            container_callback_url = runtime.rewrite_callback_url(callback_base_url)

            # --- 5b. Send POST /start to the agent ---
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{agent_url}/start",
                        json={
                            "run_id": run_id,
                            "task_id": task_key,
                            "input": input_data,
                            "callback_url": container_callback_url,
                            "agent_config": agent_config_payload,
                        },
                        timeout=10.0,
                    )
                    resp.raise_for_status()
            except Exception as exc:
                try:
                    await runtime.terminate(agent_url)
                except Exception:
                    pass
                raise RuntimeError(
                    f"[step '{step_id}'] Failed to start agent at {agent_url}: {exc}"
                ) from exc

            logger.info(
                "[step '%s'] agent started — polling run '%s' until output arrives",
                step_id, run_id,
            )

            # Write task entity to MongoDB for poll-based tracking
            if agent_task_repository is not None:
                from datetime import datetime, timezone
                _task_doc = {
                    "_id": task_key,
                    "run_id": run_id,
                    "step_id": step_id,
                    "agent_id": agent_id,
                    "agent_url": agent_url,
                    "input": input_data,
                    "agent_config": agent_config_payload if isinstance(agent_config_payload, dict) else {},
                    "status": "pending",
                    "loop_count": 0,
                    "max_loops": get_settings().agent_max_loops,
                    "outputs": [],
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
                try:
                    await agent_task_repository.save_task(_task_doc)
                except Exception as _e:
                    logger.warning("[step '%s'] failed to save agent task: %s", step_id, _e)

            # Write PVC lease for TTL cleanup
            pvc_mount_point = step.get("pvc_mount_point")
            if pvc_mount_point and pvc_lease_repository is not None:
                from datetime import datetime, timezone
                from app.runtime.pvc_manager import parse_ttl
                pvc_name = step.get("pvc_name") or f"pvc-{run_id[:12]}"
                ttl = parse_ttl(step.get("pvc_ttl", "1h"))
                expires_at = datetime.now(timezone.utc) + ttl
                lease = {
                    "pvc_name": pvc_name,
                    "namespace": settings.agent_namespace if settings else "default",
                    "run_id": run_id,
                    "expires_at": expires_at,
                    "created_at": datetime.now(timezone.utc),
                }
                try:
                    await pvc_lease_repository.save(lease)
                except Exception as _exc:
                    logger.warning("Failed to save PVC lease for %s: %s", pvc_name, _exc)

        else:
            # Retrieve real agent_url from task repository for resume path
            _stored_task = None
            if agent_task_repository is not None:
                _stored_task = await agent_task_repository.get_task(task_key)
            if _stored_task and _stored_task.get("agent_url"):
                agent_url = _stored_task["agent_url"]
            else:
                # Fall back to runtime lookup (only if method is a coroutine function)
                import inspect as _inspect
                _get_url_fn = getattr(runtime, "get_agent_url_for_run", None)
                agent_url = await _get_url_fn(agent_def, run_id) if _get_url_fn is not None and _inspect.iscoroutinefunction(_get_url_fn) else None
            if not agent_url:
                # Stale pod from a failed spawn (e.g. helm timeout before task was saved).
                # Terminate it and fall through to a fresh spawn below.
                logger.warning(
                    "[step '%s'] agent_url not found for run '%s' — stale pod detected, terminating and respawning",
                    step_id, run_id,
                )
                _terminate_fn = getattr(runtime, "terminate_by_run_id", None)
                if _terminate_fn is not None:
                    try:
                        await _terminate_fn(agent_def, run_id)
                    except Exception as _te:
                        logger.warning("[step '%s'] terminate_by_run_id failed: %s", step_id, _te)
                _is_resume = False
                agent_url = await runtime.spawn(agent_def, step, run_id, callback_base_url, extra_env=resolved_env_vars)
                logger.info("[step '%s'] agent server respawned at %s", step_id, agent_url)
                container_callback_url = runtime.rewrite_callback_url(callback_base_url)
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(
                            f"{agent_url}/start",
                            json={
                                "run_id": run_id,
                                "task_id": task_key,
                                "input": input_data,
                                "callback_url": container_callback_url,
                                "agent_config": agent_config_payload,
                            },
                            timeout=10.0,
                        )
                        resp.raise_for_status()
                except Exception as exc:
                    try:
                        await runtime.terminate(agent_url)
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"[step '{step_id}'] Failed to start respawned agent at {agent_url}: {exc}"
                    ) from exc
                if agent_task_repository is not None:
                    from datetime import datetime, timezone
                    _task_doc = {
                        "_id": task_key,
                        "run_id": run_id,
                        "step_id": step_id,
                        "agent_id": agent_id,
                        "agent_url": agent_url,
                        "input": input_data,
                        "agent_config": agent_config_payload if isinstance(agent_config_payload, dict) else {},
                        "status": "pending",
                        "loop_count": 0,
                        "max_loops": get_settings().agent_max_loops,
                        "outputs": [],
                        "created_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc),
                    }
                    try:
                        await agent_task_repository.save_task(_task_doc)
                    except Exception as _e:
                        logger.warning("[step '%s'] failed to save agent task on respawn: %s", step_id, _e)
            else:
                logger.info("[step '%s'] resuming run '%s' (container already running)", step_id, run_id)

        # --- 5c. Pull-based polling: poll agent every N seconds until finished/failed/idle ---
        _poll_interval = get_settings().agent_poll_interval_seconds
        _max_loops = get_settings().agent_max_loops
        _loop_count = 0
        raw_output: dict = {}

        while True:
            await asyncio.sleep(_poll_interval)
            try:
                async with httpx.AsyncClient() as _hc:
                    _poll_resp = await _hc.get(f"{agent_url}/poll", timeout=10.0)
                    _poll_resp.raise_for_status()
                    _poll_data = _poll_resp.json()
            except Exception as _poll_exc:
                logger.warning("[step '%s'] poll failed: %s", step_id, _poll_exc)
                _loop_count += 1
                if _loop_count >= _max_loops and agent_task_repository is not None:
                    await agent_task_repository.update_task(task_key, {"status": "failed"})
                if _loop_count >= _max_loops:
                    raise RuntimeError(f"[step '{step_id}'] agent unreachable after {_max_loops} poll attempts") from _poll_exc
                continue

            _poll_status = _poll_data.get("status", "idle")
            _poll_outputs = _poll_data.get("outputs", [])

            if agent_task_repository is not None and _poll_outputs:
                try:
                    await agent_task_repository.append_outputs(task_key, _poll_outputs)
                except Exception as _e:
                    logger.warning("[step '%s'] failed to append agent task outputs: %s", step_id, _e)

            # Cache final output from every poll cycle — the "final" output may arrive
            # while status is still "working" (task_store marks it sent before the
            # in-memory state["status"] flips to "done"), so we must capture it here
            # rather than waiting until the "finished" poll cycle (which may see outputs=[]).
            if not raw_output:
                for _out in _poll_outputs:
                    if _out.get("type") == "final":
                        raw_output = _out.get("content", {})
                        break

            # Forward progress messages to run state so the UI can see them.
            if run_repository is not None and _poll_outputs:
                _progress_msgs = [
                    out["content"]["message"]
                    for out in _poll_outputs
                    if out.get("type") == "progress"
                    and isinstance(out.get("content"), dict)
                    and out["content"].get("message")
                    and not out["content"]["message"].startswith("__")
                ]
                if _progress_msgs:
                    try:
                        _run = await run_repository.get(run_id)
                        if _run is not None:
                            _progress_key = f"_agent_progress_{step_id}"
                            _progress_list = list((_run.state or {}).get(_progress_key, []))
                            _progress_list.extend(_progress_msgs)
                            _run.state = {**(_run.state or {}), _progress_key: _progress_list}
                            _run.touch()
                            await run_repository.update(_run)
                    except Exception:
                        pass  # progress is best-effort

            if _poll_status == "finished":
                # raw_output may already be set from an earlier poll cycle (race condition
                # where "final" output was consumed before status flipped to "finished").
                # Fall back to scanning _poll_outputs only if not already captured.
                if not raw_output:
                    for _out in reversed(_poll_outputs):
                        if _out.get("type") == "final":
                            raw_output = _out.get("content", {})
                            break
                if not raw_output:
                    # Final output may have been consumed by a previous poll cycle or
                    # missed due to a race — check the task repository as fallback.
                    if agent_task_repository is not None:
                        _stored = await agent_task_repository.get_task(task_key)
                        if _stored:
                            for _out in reversed(_stored.get("outputs", [])):
                                if isinstance(_out, dict) and _out.get("type") == "final":
                                    raw_output = _out.get("content", {})
                                    logger.info("[step '%s'] recovered final output from task repository", step_id)
                                    break
                    if not raw_output:
                        logger.warning("[step '%s'] finished status but no 'final' output found in poll outputs or task repository", step_id)
                if agent_task_repository is not None:
                    await agent_task_repository.update_task(task_key, {"status": "finished"})
                break

            if _poll_status == "failed":
                if agent_task_repository is not None:
                    await agent_task_repository.update_task(task_key, {"status": "failed"})
                raise RuntimeError(f"[step '{step_id}'] agent reported failure")

            if _poll_status == "idle":
                # Agent lost the task — meta LLM recovery
                if use_meta_llm:
                    from app.services.agent_poller import _meta_llm_recovery
                    _task_for_recovery = {"input": input_data, "outputs": _poll_outputs}
                    if agent_task_repository is not None:
                        _stored_task = await agent_task_repository.get_task(task_key)
                        if _stored_task:
                            _task_for_recovery = _stored_task
                    _is_complete = await _meta_llm_recovery(_task_for_recovery, settings)
                    if _is_complete:
                        for _out in reversed(_task_for_recovery.get("outputs", [])):
                            if isinstance(_out, dict) and _out.get("type") == "final":
                                raw_output = _out.get("content", {})
                                break
                        if agent_task_repository is not None:
                            await agent_task_repository.update_task(task_key, {"status": "finished"})
                        break
                _loop_count += 1
                if _loop_count >= _max_loops:
                    if agent_task_repository is not None:
                        await agent_task_repository.update_task(task_key, {"status": "failed"})
                    raise RuntimeError(f"[step '{step_id}'] agent idle after {_max_loops} recovery attempts")
                # Resend task
                logger.info("[step '%s'] resending task to agent (loop %d/%d)", step_id, _loop_count, _max_loops)
                if agent_task_repository is not None:
                    await agent_task_repository.update_task(task_key, {"loop_count": _loop_count})
                try:
                    async with httpx.AsyncClient() as _hc2:
                        _restart_resp = await _hc2.post(
                            f"{agent_url}/start",
                            json={"run_id": run_id, "task_id": task_key, "input": input_data, "callback_url": container_callback_url, "agent_config": agent_config_payload},
                            timeout=10.0,
                        )
                        _restart_resp.raise_for_status()
                except Exception as _restart_exc:
                    logger.warning("[step '%s'] resend failed: %s", step_id, _restart_exc)
                continue
            # status == "working" or unknown — keep polling

        # After poll loop: either park as warm pod or terminate.
        if warm_pod_repository is not None and runtime_type == "k8s":
            from datetime import datetime, timezone, timedelta
            from app.infrastructure.persistence.mongo import WarmPodRecord
            _now = datetime.now(timezone.utc)
            _release_name = runtime._release_name(agent_def, run_id) if hasattr(runtime, "_release_name") else ""
            try:
                await warm_pod_repository.upsert(WarmPodRecord(
                    run_id=run_id,
                    agent_id=agent_id,
                    agent_url=agent_url,
                    release_name=_release_name,
                    created_at=_now,
                    expires_at=_now + timedelta(hours=24),
                ))
                logger.info("[step '%s'] parked warm pod at %s (release=%s)", step_id, agent_url, _release_name)
            except Exception as _wp_exc:
                logger.warning("[step '%s'] failed to upsert warm pod record: %s — terminating instead", step_id, _wp_exc)
                if hasattr(runtime, "terminate_by_run_id"):
                    await runtime.terminate_by_run_id(agent_def, run_id)
                else:
                    try:
                        await runtime.terminate(agent_url)
                    except Exception as _exc:
                        logger.warning("[step '%s'] failed to terminate agent: %s", step_id, _exc)
        else:
            # Terminate using run_id label — works whether _containers is populated or not.
            if hasattr(runtime, "terminate_by_run_id"):
                await runtime.terminate_by_run_id(agent_def, run_id)
            else:
                try:
                    await runtime.terminate(agent_url)
                except Exception as _exc:
                    logger.warning("[step '%s'] failed to terminate agent: %s", step_id, _exc)

        logger.info(
            "[step '%s'] agent run '%s' resumed, output keys: %s",
            step_id, run_id, list(raw_output),
        )

        # --- 5d-pre. Extract structured output from free-form "result" text ---
        # Agent pod frameworks wrap Claude's response in {"result": "...", "token_usage": {...}}
        # even when the Output Protocol or system prompt instructed Claude to return structured
        # data.  When the step has output_mapping and the raw output is just {"result": "text"},
        # try to parse that text as JSON or YAML so the structured fields reach output_mapping.
        _pre_output_mapping = step.get("output_mapping") or {}
        _expected_keys = set(_pre_output_mapping.keys())
        if (
            _expected_keys
            and isinstance(raw_output.get("result"), str)
            and not any(k in raw_output for k in _expected_keys)
        ):
            import json as _json, re as _re
            _result_text = raw_output["result"].strip()

            def _try_parse(text: str) -> dict | None:
                """Try JSON then YAML; return dict if any expected key found."""
                # JSON attempt
                try:
                    _p = _json.loads(text)
                    if isinstance(_p, dict) and any(k in _p for k in _expected_keys):
                        return _p
                except Exception:
                    pass
                # YAML attempt (agent system prompts often instruct YAML output)
                try:
                    import yaml as _yaml
                    _p = _yaml.safe_load(text)
                    if isinstance(_p, dict) and any(k in _p for k in _expected_keys):
                        return _p
                except Exception:
                    pass
                return None

            # Try: yaml/json code fence, then raw text
            _parsed_out: dict | None = None
            _fence = _re.search(
                r"```(?:yaml|json)?\s*\n?(.*)\n?\s*```", _result_text, _re.DOTALL | _re.IGNORECASE
            )
            if _fence:
                _parsed_out = _try_parse(_fence.group(1).strip())
            if _parsed_out is None:
                _parsed_out = _try_parse(_result_text)

            if _parsed_out is None:
                # Log the raw result so we can diagnose why extraction failed.
                logger.warning(
                    "[step '%s'] could not extract structured output from 'result' text "
                    "(output_mapping has %d fields). Raw result (first 500 chars): %r",
                    step_id, len(_pre_output_mapping), _result_text[:500],
                )
            else:
                logger.info(
                    "[step '%s'] extracted structured output from 'result' text "
                    "(%d/%d output_mapping fields matched)",
                    step_id,
                    sum(1 for k in _pre_output_mapping if k in _parsed_out),
                    len(_pre_output_mapping),
                )
                raw_output = {
                    **_parsed_out,
                    **{k: v for k, v in raw_output.items() if k not in _parsed_out and k != "result"},
                }

        # --- 5d. Surface unanswered clarify-tool question, then meta-LLM ---
        # If the agent called the clarify tool but nobody answered (timeout or skip),
        # _pending_question is still set in the run's DB state. Re-surface it as an
        # ask_context interrupt so the UI and Slack can handle it properly — bypassing
        # meta-LLM which would otherwise decide "proceed" on the partial output.
        _surfaced_pending = False

        # Deterministic context-sufficiency gate — no LLM needed.
        # Ideally agents use the clarify tool mid-run so they always exit with
        # context_sufficient=True.  This gate is a safety net for agents that
        # still return context_sufficient=False in their final output.
        if not raw_output.get("context_sufficient", True):
            questions = raw_output.get("questions", [])
            if isinstance(questions, list) and questions:
                logger.warning(
                    "[step '%s'] context_sufficient=False in final output — agent should use "
                    "the clarify tool instead of exiting; surfacing %d question(s): %s",
                    step_id, len(questions), questions,
                )
                answers = interrupt({"type": "ask_context", "questions": questions})
                if isinstance(answers, dict) and answers:
                    raw_output = {**raw_output, "_clarification_answers": answers}
                _surfaced_pending = True

        if run_repository is not None:
            try:
                fresh_run = await run_repository.get(run_id)
                pending_q = (fresh_run.state or {}).get("_pending_question") if fresh_run else None
            except Exception:
                pending_q = None
            if pending_q:
                question_text = (
                    pending_q.get("question", str(pending_q))
                    if isinstance(pending_q, dict) else str(pending_q)
                )
                logger.info(
                    "[step '%s'] unanswered clarify question detected — surfacing as ask_context",
                    step_id,
                )
                answers = interrupt({"type": "ask_context", "questions": [question_text]})
                if isinstance(answers, dict) and answers:
                    raw_output = {**raw_output, "_clarification_answers": answers}
                # Clear the marker so it doesn't re-trigger on the next resume
                try:
                    r2 = await run_repository.get(run_id)
                    if r2 and "_pending_question" in (r2.state or {}):
                        r2.state = {k: v for k, v in r2.state.items() if k != "_pending_question"}
                        r2.touch()
                        await run_repository.update(r2)
                except Exception:
                    pass
                _surfaced_pending = True

        # --- 5b. Meta-LLM step evaluation ---
        # Run BEFORE output_mapping validation so the quality gate fires even when
        # the agent returned plain text instead of structured fields.  This produces
        # a meaningful "meta-LLM rejected: <reason>" error rather than the opaque
        # "expected fields not found" message, and prevents downstream steps from
        # running on obviously bad output.
        _meta_llm_verdict: dict[str, Any] | None = None
        _meta_llm_usage: dict[str, int] | None = None
        if use_meta_llm and not _surfaced_pending:
            _sc = step.get("success_criteria") if isinstance(step, dict) else None
            _fc = step.get("fail_criteria") if isinstance(step, dict) else None
            logger.info(
                "[step '%s'] running meta-LLM evaluation (success_criteria=%r, fail_criteria=%r)",
                step_id, _sc, _fc,
            )
            _eval = await _meta_llm_evaluate(raw_output, input_data, step_id, settings, _sc, _fc)
            logger.info(
                "[step '%s'] meta-LLM result: passed=%s reason=%r",
                step_id, _eval["passed"], _eval.get("reason"),
            )
            _meta_llm_usage = _eval.get("usage")
            # Store the meta-LLM verdict regardless of pass/fail so the UI can
            # always show what the quality gate decided and why.
            _meta_llm_verdict = {
                "passed": _eval["passed"],
                "reason": _eval.get("reason", ""),
            }
            if not _eval["passed"]:
                _rejection_reason = _eval["reason"]
                _rej_output_mapping: dict[str, str] | None = step.get("output_mapping")
                _rej_output_key: str | None = step.get("output_key")
                if _rej_output_mapping:
                    _mapped_for_rejection: dict[str, Any] = {
                        wk: raw_output[ak]
                        for ak, wk in _rej_output_mapping.items()
                        if ak in raw_output
                    }
                elif _rej_output_key and "result" in raw_output:
                    _mapped_for_rejection = {_rej_output_key: raw_output["result"]}
                else:
                    _mapped_for_rejection = {}
                _apply_usage_keys(_mapped_for_rejection, step_id, raw_output, _meta_llm_usage)
                raise MetaLLMRejectionError(
                    _rejection_reason,  # clean reason — no step-id prefix noise
                    mapped_result=_mapped_for_rejection,
                    reason=_rejection_reason,
                )

        # Deterministic fail when output_mapping is set but nothing matched.
        # The agent returned unstructured output (e.g. {"result": "text"}) AND
        # YAML/JSON extraction above couldn't find any expected fields.
        # Check for empty/minimal output separately — that's an agent execution
        # failure (max iterations, tool errors), not a contract violation.
        _output_mapping_check = step.get("output_mapping") or {}
        if (
            not _surfaced_pending
            and _output_mapping_check
            and not any(k in raw_output for k in _output_mapping_check)
            and "context_sufficient" not in raw_output
        ):
            _result_val = raw_output.get("result", "")
            _raw_snippet = str(_result_val or raw_output)[:5000]
            _token_usage = raw_output.get("token_usage", {})
            _output_tokens = _token_usage.get("output_tokens", 0) if isinstance(_token_usage, dict) else 0
            # "(no output)" is LangGraph's fallback when the ReAct loop ends
            # without a final AI message (max iterations hit, context overflow, etc.)
            _is_framework_empty = (not _result_val) or _result_val.strip() in ("", "(no output)", "None", "null")
            if _is_framework_empty:
                _extra = (
                    f"Token usage suggests agent worked ({_output_tokens:,} output tokens) "
                    "but hit max iterations or context limit before producing structured output. "
                    if _output_tokens > 100 else
                    "Agent may not have run successfully. "
                )
                raise RuntimeError(
                    f"[step '{step_id}'] Agent returned no usable output. "
                    f"{_extra}"
                    f"Token usage: {_token_usage}"
                )
            # Use MetaLLMRejectionError so the UI can display the raw output alongside
            # the error — the agent may have returned a valid message (e.g. "Jira unavailable")
            # that is useful to show even though it didn't match the structured schema.
            _raw_mapped: dict[str, Any] = {}
            _apply_usage_keys(_raw_mapped, step_id, raw_output, _meta_llm_usage)
            raise MetaLLMRejectionError(
                f"Agent returned unstructured output — expected fields {list(_output_mapping_check)} not found. "
                f"Agent said: {_raw_snippet}",
                mapped_result=_raw_mapped,
                reason=f"Agent returned plain text instead of structured output. Agent said: {_raw_snippet}",
            )

    # Fail fast when the agent returned {"error": "...", "token_usage": {...}} with no
    # "result" key — this is the framework's error format (max iterations, API error, etc.).
    # Without this check, the error dict would be stored as the output_key value and the
    # workflow would continue with garbage data.
    if not _surfaced_pending and isinstance(raw_output, dict):
        _agent_error_msg = raw_output.get("error")
        if _agent_error_msg and not raw_output.get("result"):
            raise RuntimeError(
                f"[step '{step_id}'] Agent reported error: {_agent_error_msg}. "
                f"Token usage: {raw_output.get('token_usage', {})}"
            )

    # --- 6. Map output back to workflow state ---
    output_mapping: dict[str, str] | None = step.get("output_mapping")
    output_key: str | None = step.get("output_key")

    _final_meta_llm_usage = locals().get("_meta_llm_usage")

    if output_mapping:
        # Map individual agent output keys back to workflow state keys.
        # {agent_key: workflow_key}
        result: dict[str, Any] = {
            workflow_key: raw_output[agent_key]
            for agent_key, workflow_key in output_mapping.items()
            if agent_key in raw_output
        }
        # Always preserve token usage regardless of output_mapping declaration.
        # Agent-LLM, post-compact meta-LLM, and backend judge usage are kept as
        # three separate buckets — never merged into one another.
        _apply_usage_keys(result, step_id, raw_output, _final_meta_llm_usage)
    elif output_key:
        if isinstance(raw_output, dict) and "result" in raw_output:
            # Agent sent {"result": "...", "token_usage": {...}} — extract result key
            result = {output_key: raw_output["result"]}
            _apply_usage_keys(result, step_id, raw_output, _final_meta_llm_usage)
        elif isinstance(raw_output, dict) and len(raw_output) == 1:
            result = {output_key: next(iter(raw_output.values()))}
            _apply_usage_keys(result, step_id, raw_output, _final_meta_llm_usage)
        else:
            result = {output_key: raw_output}
    else:
        # No mapping configured — merge all agent output keys directly into state.
        result = raw_output

    # Always surface the meta-LLM verdict so the UI shows full transparency on
    # every step — not just failed ones. The agent's full raw output (including
    # every tool invocation) is already tracked via _agent_progress; no need to
    # duplicate the final result text here.
    if _meta_llm_verdict is not None:
        result["_meta_llm_result"] = _meta_llm_verdict

    return result
