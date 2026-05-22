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


def _build_agent_config(
    agent_def: "AgentDefinition",
    settings: "Settings",
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
        mcp_servers.append(entry)

    # --- Credentials (resolved API-key values) ---
    credentials: dict[str, str] = {}
    for llm_intg in settings.get_llm_integrations():
        key = llm_intg.resolved_api_key()
        if key:
            credentials[llm_intg.resolved_api_key_env()] = key

    return {
        "system_prompt": system_prompt,
        "model": model,
        "tools": tools,
        "mcp_servers": mcp_servers,
        "credentials": credentials,
        "extra": agent_input,
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


async def execute_agent_step(
    step: dict[str, Any],
    state: dict[str, Any],
    agent_backend: "AgentDefinitionBackend",
    run_id: str,
    callback_base_url: str,
    settings: "Settings | None" = None,
    progress_cb: Callable[[str], Awaitable[None]] | None = None,
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
        )
        logger.info("[step '%s'] local agent completed, output keys: %s", step_id, list(raw_output))
    else:
        # Docker / K8s: use the HTTP protocol with interrupt-based suspension.
        from app.runtime.factory import get_runtime

        runtime: AgentRuntime = get_runtime(runtime_type)
        agent_config_payload = _build_agent_config(agent_def, settings)

        # --- 5a. Spawn the agent HTTP server ---
        agent_url = await runtime.spawn(agent_def, step, run_id, callback_base_url)
        logger.info("[step '%s'] agent server spawned at %s", step_id, agent_url)

        # --- 5b. Send POST /start to the agent ---
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{agent_url}/start",
                    json={
                        "run_id": run_id,
                        "input": input_data,
                        "callback_url": callback_base_url,
                        "agent_config": agent_config_payload,
                    },
                    timeout=10.0,
                )
                resp.raise_for_status()
        except Exception as exc:
            # If /start fails, terminate the agent and propagate the error.
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

        # --- 5c. Suspend: interrupt() pauses the LangGraph node ---
        # The run transitions to "waiting_agent" status (handled in stream_graph_to_pause).
        # Resume comes from POST /api/v1/runs/{run_id}/agent/output via
        # Command(resume={"output": raw_output}).
        resume_value: dict = interrupt({"type": "waiting_agent", "agent_url": agent_url})
        raw_output = resume_value.get("output", {})
        logger.info(
            "[step '%s'] agent run '%s' resumed, output keys: %s",
            step_id, run_id, list(raw_output),
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
    elif output_key:
        # Store the whole output dict under a single state key.
        result = {output_key: raw_output}
    else:
        # No mapping configured — merge all agent output keys directly into state.
        result = raw_output

    return result
