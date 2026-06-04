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

import os

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import httpx
from langgraph.types import interrupt

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.domain.models.agent_definition import AgentDefinition
    from app.infrastructure.persistence.agent_backend import AgentDefinitionBackend
    from app.runtime.base import AgentRuntime

logger = logging.getLogger(__name__)

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


def _build_agent_config(
    agent_def: "AgentDefinition",
    settings: "Settings",
    step: dict[str, Any] | None = None,
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
    - ``extra``: the entire ``agent_input`` dict forwarded as-is.
    - ``description`` is NOT included.
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

    # Apply compression level from step config (prepend instruction to system prompt)
    compression_level = (step or {}).get("compression_level", "none")
    compression_instruction = _COMPRESSION_INSTRUCTIONS.get(compression_level or "none", "")
    if compression_instruction:
        system_prompt = (
            f"{compression_instruction}\n\n{system_prompt}"
            if system_prompt
            else compression_instruction
        )

    # Inject Output Protocol when the step declares an output_mapping.
    output_mapping = (step or {}).get("output_mapping") or {}
    if output_mapping:
        field_list = "\n".join(f"- {k}" for k in output_mapping)
        protocol = (
            "\n\n## Output Protocol\n\n"
            "Your output MUST include these fields (YAML or JSON — the orchestrator "
            "extracts either format):\n\n"
            f"{field_list}"
        )
        system_prompt = f"{system_prompt}{protocol}" if system_prompt else protocol.lstrip()

    # --- MCP servers ---
    raw_integrations: list[McpIntegrationConfig] = settings.get_mcp_integrations()
    allowed_tools: set[str] | None = (
        set(tools) if tools is not None else None
    )
    mcp_servers: list[dict[str, Any]] = []
    for intg in raw_integrations:
        if allowed_tools is not None and intg.name not in allowed_tools:
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

    return {
        "system_prompt": system_prompt,
        "model": model,
        "tools": tools,
        "mcp_servers": mcp_servers,
        "credentials": credentials,
        "extra": extra,
        "env_vars": env_vars,
    }


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


async def _meta_llm_decide(
    raw_output: dict,
    input_data: dict,
    step_id: str,
    settings: "Settings",
) -> dict:
    """Call a lightweight LLM to decide how to proceed after an agent step.

    Returns: {"decision": "proceed"|"ask_clarification"|"ask_approval",
              "questions": list[str], "reason": str}
    Always returns proceed on any failure (non-blocking).
    """
    import json as _json
    try:
        from app.core.container import build_llm_native
        provider = settings.meta_llm_provider or settings.llm_provider
        model = settings.meta_llm_model
        llm = build_llm_native(provider, model, settings, max_tokens=512)

        request_text = (
            input_data.get("request")
            or input_data.get("task")
            or input_data.get("prompt")
            or str(input_data)
        )
        output_text = (
            raw_output.get("result") or raw_output.get("answer") or str(raw_output)
            if isinstance(raw_output, dict) else str(raw_output)
        )

        prompt = (
            "You are an orchestrator analyzing an AI agent's response.\n\n"
            f"Original request: {request_text}\n\n"
            f"Agent output:\n{output_text}\n\n"
            "Decide the next action:\n"
            "1. Agent successfully answered → PROCEED\n"
            "2. Agent needs more context that the user can provide → ASK_CLARIFICATION (extract the questions)\n"
            "3. Output needs human review before continuing → ASK_APPROVAL\n"
            "4. Agent cannot proceed due to missing tools/credentials/system access → FAIL\n\n"
            "Use FAIL when the blocker is a hard configuration issue: "
            "gcloud not authenticated, missing API key/token, tool not installed, "
            "no access to required system. These cannot be resolved by answering questions.\n"
            "Use ASK_CLARIFICATION when the user could unblock the agent by providing "
            "information (e.g. which project, which environment, what format).\n\n"
            "Respond with ONLY (no preamble):\n"
            "DECISION: <PROCEED|ASK_CLARIFICATION|ASK_APPROVAL|FAIL>\n"
            "QUESTIONS: [\"q1\", \"q2\"]  (only when ASK_CLARIFICATION)\n"
            "REASON: <one sentence>"
        )

        from langchain_core.messages import HumanMessage
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        text = response.content if isinstance(response.content, str) else str(response.content)

        decision = "proceed"
        questions: list[str] = []
        reason = ""

        for line in text.strip().splitlines():
            line = line.strip()
            if line.startswith("DECISION:"):
                raw_d = line[len("DECISION:"):].strip().upper()
                if raw_d == "ASK_CLARIFICATION":
                    decision = "ask_clarification"
                elif raw_d == "ASK_APPROVAL":
                    decision = "ask_approval"
                elif raw_d == "FAIL":
                    decision = "fail"
                else:
                    decision = "proceed"
            elif line.startswith("QUESTIONS:"):
                raw_q = line[len("QUESTIONS:"):].strip()
                try:
                    parsed = _json.loads(raw_q)
                    if isinstance(parsed, list):
                        questions = [str(q) for q in parsed]
                except Exception:
                    # LLM returned non-JSON (e.g. plain text). Treat the
                    # whole raw value as a single question so it is never lost.
                    if raw_q:
                        questions = [raw_q]
            elif line.startswith("REASON:"):
                reason = line[len("REASON:"):].strip()

        if decision == "ask_clarification" and not questions:
            # LLM decided clarification is needed but produced no parseable
            # QUESTIONS line. Log the full response so we can diagnose the
            # LLM output format; the caller will use reason as a fallback.
            logger.warning(
                "[step '%s'] meta-LLM ask_clarification with no parseable questions. "
                "Full response: %r",
                step_id, text,
            )

        logger.info("[step '%s'] meta-LLM decision: %s — %s", step_id, decision, reason)
        return {"decision": decision, "questions": questions, "reason": reason}

    except Exception as exc:
        logger.warning("[step '%s'] meta-LLM analysis failed: %s — proceeding normally", step_id, exc)
        return {"decision": "proceed", "questions": [], "reason": str(exc)}


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

    # --- 3. Resolve settings (lazy-import to avoid circular imports at module load) ---
    if settings is None:
        from app.core.config import get_settings
        settings = get_settings()

    # --- 4. Build input from state via input_mapping ---
    input_mapping: dict[str, str] | None = step.get("input_mapping")
    input_data: dict[str, Any] = _apply_mapping(state, input_mapping)
    # If a previous ask_context interrupt was answered, fold those answers into
    # the input so the agent can use them as clarifying context.
    if state.get("_clarification_answers"):
        input_data = {**input_data, "clarification_context": state["_clarification_answers"]}

    # --- 5. Branch on runtime ---
    if runtime_type == "local":
        # Run inline — no HTTP, no subprocess.
        from app.agents.local_agent import run_local_agent

        logger.info("[step '%s'] running local inline agent", step_id)
        raw_output = await run_local_agent(
            agent_input=agent_def.agent_input,
            input_data=input_data,
            settings=settings,
            progress_cb=progress_cb,
            compression_level=step.get("compression_level", "none"),
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
        agent_config_payload = _build_agent_config(agent_def, settings, step=step)
        resolved_env_vars: dict[str, str] = agent_config_payload.get("env_vars") or {}

        # LangGraph reruns the node from scratch on resume. Detect this by checking
        # whether a container already exists for this run (spawned in the first execution).
        # If so, skip spawn+start — interrupt() will return immediately with the stored output.
        _is_resume = (
            hasattr(runtime, "has_container_for_run")
            and await runtime.has_container_for_run(run_id)
        )

        if not _is_resume:
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
                "[step '%s'] agent started — suspending run '%s' until output arrives",
                step_id, run_id,
            )

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
            agent_url = f"<resumed:{run_id}>"
            logger.info("[step '%s'] resuming run '%s' (container already running)", step_id, run_id)

        # --- 5c. Suspend: interrupt() pauses the LangGraph node ---
        # On first execution: suspends until POST /agent/output resumes the graph.
        # On resume (LangGraph reruns node): returns immediately with stored output.
        resume_value: dict = interrupt({"type": "waiting_agent", "agent_url": agent_url})
        raw_output = resume_value.get("output", {})

        # Terminate using run_id label — works whether _containers is populated or not.
        if hasattr(runtime, "terminate_by_run_id"):
            await runtime.terminate_by_run_id(run_id)
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
        if (
            _pre_output_mapping
            and isinstance(raw_output.get("result"), str)
            and not any(k in raw_output for k in _pre_output_mapping)
        ):
            import json as _json, re as _re
            _result_text = raw_output["result"].strip()

            def _try_parse(text: str) -> dict | None:
                """Try JSON then YAML; return dict if any output_mapping key found."""
                # JSON attempt
                try:
                    _p = _json.loads(text)
                    if isinstance(_p, dict) and any(k in _p for k in _pre_output_mapping):
                        return _p
                except Exception:
                    pass
                # YAML attempt (agent system prompts often instruct YAML output)
                try:
                    import yaml as _yaml
                    _p = _yaml.safe_load(text)
                    if isinstance(_p, dict) and any(k in _p for k in _pre_output_mapping):
                        return _p
                except Exception:
                    pass
                return None

            # Try: yaml/json code fence, then raw text
            _parsed_out: dict | None = None
            _fence = _re.search(
                r"```(?:yaml|json)?\s*\n?(.*?)\n?\s*```", _result_text, _re.DOTALL | _re.IGNORECASE
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
            _raw_snippet = str(_result_val or raw_output)[:400]
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
            raise RuntimeError(
                f"[step '{step_id}'] Agent returned unstructured output — "
                f"expected fields {list(_output_mapping_check)} not found. "
                f"Raw output (truncated): {_raw_snippet}"
            )

    # --- 6. Map output back to workflow state ---
    output_mapping: dict[str, str] | None = step.get("output_mapping")
    output_key: str | None = step.get("output_key")

    if output_mapping:
        # Map individual agent output keys back to workflow state keys.
        # {agent_key: workflow_key}
        result: dict[str, Any] = {
            workflow_key: raw_output[agent_key]
            for agent_key, workflow_key in output_mapping.items()
            if agent_key in raw_output
        }
        # Always preserve token usage regardless of output_mapping declaration.
        if "token_usage" in raw_output:
            result[f"_agent_token_usage_{step_id}"] = raw_output["token_usage"]
    elif output_key:
        if isinstance(raw_output, dict) and "result" in raw_output:
            # Agent sent {"result": "...", "token_usage": {...}} — extract result key
            result = {output_key: raw_output["result"]}
            if "token_usage" in raw_output:
                result[f"_agent_token_usage_{step_id}"] = raw_output["token_usage"]
        elif isinstance(raw_output, dict) and len(raw_output) == 1:
            result = {output_key: next(iter(raw_output.values()))}
        else:
            result = {output_key: raw_output}
    else:
        # No mapping configured — merge all agent output keys directly into state.
        result = raw_output

    return result
