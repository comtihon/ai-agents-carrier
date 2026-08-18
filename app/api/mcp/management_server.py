"""FastMCP server exposing the platform's own management tools at /mcp/management.

Same tool set as the internal chat agent — workflow/agent/data-source CRUD plus
run control — minus ``ask_user`` (which only makes sense inside a chat turn).
Both surfaces call the shared cores in ``app.application.management_tools`` and
``app.application.run_control``, so there is one implementation per operation.

Handlers resolve the container at *call* time (like the datasources server), so
registration can happen before the container exists.  The tool set is static,
so it is registered once at startup and never rebuilt.

Approver identity: MCP callers have no JWT claims channel, so approvals and
rejections made here are recorded as ``approver_name="management-mcp"``,
``approver_id=None``, ``approver_source="mcp"``.  The endpoint itself is
token-gated, which is what the audit trail relies on.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from app.application import management_tools as core
from app.application import run_control
from app.application.management_tools import _spawn_background, deps_from_container
from app.application.run_control import RunControlError

logger = logging.getLogger(__name__)

# Mounted at "/mcp" by create_app, so the full endpoint is /mcp/management.
STREAMABLE_HTTP_PATH = "/management"


APPROVER_NAME = "management-mcp"
APPROVER_SOURCE = "mcp"

# FastMCP's DNS-rebinding guard rejects any Host header it was not told about
# with 421.  Its own default requires a port, so even "Host: localhost" fails.
# These defaults keep local development working; a deployment reached under a
# real hostname must add it via MANAGEMENT_MCP_ALLOWED_HOSTS (never "*").
DEFAULT_ALLOWED_HOSTS = [
    "localhost",
    "localhost:*",
    "127.0.0.1",
    "127.0.0.1:*",
    "[::1]",
    "[::1]:*",
]

_mcp: FastMCP | None = None


def _transport_security(
    allowed_hosts: list[str] | None = None,
) -> TransportSecuritySettings:
    """DNS-rebinding settings for the management endpoint.

    Origins are derived from the same host list so a browser-based MCP client
    sending ``Origin`` is not rejected for a host that is explicitly allowed.
    """
    hosts = [h for h in (allowed_hosts or DEFAULT_ALLOWED_HOSTS) if h]
    origins = [f"{scheme}://{host}" for host in hosts for scheme in ("http", "https")]
    return TransportSecuritySettings(allowed_hosts=hosts, allowed_origins=origins)


def build_management_mcp(allowed_hosts: list[str] | None = None) -> FastMCP:
    """Create the (stateless) FastMCP server instance for platform management."""
    return FastMCP(
        "management",
        instructions=(
            "Manage this workflow automation platform: list/create/update/delete "
            "workflows, agent definitions and data sources, start workflow runs, "
            "inspect them, and control them (terminate, retry, restart from a "
            "step, approve, reject)."
        ),
        stateless_http=True,
        streamable_http_path=STREAMABLE_HTTP_PATH,
        transport_security=_transport_security(allowed_hosts),
    )


def get_management_mcp(allowed_hosts: list[str] | None = None) -> FastMCP:
    """Return the process-wide management MCP server, building it on demand.

    ``allowed_hosts`` only applies on the first (building) call.
    """
    global _mcp
    if _mcp is None:
        _mcp = build_management_mcp(allowed_hosts)
    return _mcp


def _refresher(container: Any) -> Callable[[], Any]:
    """Data source MCP tool refresher for the given container."""
    from app.api.app import _make_datasources_refresher
    return _make_datasources_refresher(container)


def register_management_tools(
    mcp: FastMCP, container_getter: Callable[[], Any]
) -> None:
    """Register the full management tool set on *mcp* (idempotent per server)."""

    def deps():
        container = container_getter()
        return deps_from_container(container, _refresher(container))

    # ── workflows / runs ────────────────────────────────────────────────────

    async def list_workflows() -> str:
        """List all available workflow IDs, names, and descriptions."""
        return core.list_workflows(deps())

    async def run_workflow(workflow_id: str, request: str) -> str:
        """Start a workflow run.

        Args:
            workflow_id: The workflow ID (from list_workflows).
            request: A detailed description of the task to execute.

        Returns:
            JSON with run_id, workflow_id, workflow_name, and __event__ = workflow_started.
        """
        return await core.run_workflow(deps(), workflow_id, request)

    async def list_runs(workflow_id: str | None = None, limit: int = 10) -> str:
        """List recent workflow runs.

        Args:
            workflow_id: Filter to a specific workflow (optional).
            limit: Maximum number of runs to return (default 10).
        """
        return await core.list_runs(deps(), workflow_id, limit)

    async def get_run(run_id: str) -> str:
        """Get detailed status and step-level output for a specific workflow run.

        Args:
            run_id: The run ID to inspect.
        """
        return await core.get_run(deps(), run_id)

    async def create_workflow(
        workflow_id: str, name: str, description: str, steps_json: str
    ) -> str:
        """Create a new workflow definition and register it immediately.

        Args:
            workflow_id: Unique kebab-case identifier (e.g. "send-slack-report").
            name: Human-readable display name.
            description: What this workflow does.
            steps_json: JSON array of step objects. Each step must have "id" and "type".
                Supported types: http (webhook trigger), cron (scheduled trigger),
                llm_structured (LLM with structured output), llm (free-form LLM),
                mcp (single MCP tool call), human_approval (pause for approval),
                execute (OpenHands code execution), workflow (child workflow),
                http_call (outbound HTTP), langgraph-agent, claude-agent,
                data_source (invoke a DataSourceDefinition operation).
                Example: [{"id": "trigger", "type": "http"}, {"id": "research", "type": "llm_structured", "system_prompt": "...", "output": [{"name": "summary", "type": "str", "description": "..."}]}]
        """
        return await core.create_workflow(
            deps(), workflow_id, name, description, steps_json
        )

    async def update_workflow(
        workflow_id: str,
        name: str | None = None,
        description: str | None = None,
        steps_json: str | None = None,
    ) -> str:
        """Update an existing workflow definition (name, description, and/or steps).

        Args:
            workflow_id: The workflow ID to update.
            name: New display name (omit to keep current).
            description: New description (omit to keep current).
            steps_json: JSON array replacing ALL steps (omit to keep current).
        """
        return await core.update_workflow(
            deps(), workflow_id, name, description, steps_json
        )

    async def delete_workflow(workflow_id: str) -> str:
        """Permanently delete a workflow definition.

        Args:
            workflow_id: The workflow ID to delete.
        """
        return await core.delete_workflow(deps(), workflow_id)

    # ── agent definitions ───────────────────────────────────────────────────

    async def list_agents() -> str:
        """List all available agent definitions."""
        return await core.list_agents(deps())

    async def get_agent(agent_id: str) -> str:
        """Get full agent definition by id or name."""
        return await core.get_agent(deps(), agent_id)

    async def create_agent(
        agent_id: str,
        name: str,
        description: str = "",
        default_runtime: str = "local",
        agent_input_json: str = "{}",
    ) -> str:
        """Create a new agent definition. agent_input_json is a JSON object of default input overrides."""
        return await core.create_agent(
            deps(), agent_id, name, description, default_runtime, agent_input_json
        )

    async def update_agent(
        agent_id: str,
        name: str = None,
        description: str = None,
        default_runtime: str = None,
        agent_input_json: str = None,
    ) -> str:
        """Update an existing agent definition. Only provided fields are changed; others preserved."""
        return await core.update_agent(
            deps(), agent_id, name, description, default_runtime, agent_input_json
        )

    async def delete_agent(agent_id: str) -> str:
        """Delete an agent definition by exact id."""
        return await core.delete_agent(deps(), agent_id)

    # ── data sources ────────────────────────────────────────────────────────

    async def list_datasources() -> str:
        """List all registered data sources with their operations."""
        return await core.list_datasources(deps())

    async def create_datasource(
        source_id: str,
        name: str,
        base_url: str,
        operations_json: str,
        description: str = "",
        kind: str = "http",
        auth_json: str = "",
    ) -> str:
        """Create a data source definition and publish its MCP tools.

        Args:
            source_id: Unique kebab-case identifier (e.g. "github-api").
            name: Human-readable display name.
            base_url: Base URL of the API (GraphQL sources POST to it directly).
            operations_json: JSON array of operation objects. Each needs "name" plus
                "path" (http) or "query" (graphql), and optionally "method",
                "params" ([{name, type, required, description}]), "mapping"
                (JMESPath), "paginate", "response_schema". Templates may use
                {params.x} and {other_operation.field.path}.
            description: What this data source provides.
            kind: "http" or "graphql".
            auth_json: Optional JSON auth block carrying the secret value itself,
                e.g. {"type": "bearer", "token": "<token>"},
                {"type": "basic", "username": "...", "password": "..."} or
                {"type": "header", "header_name": "X-Api-Key", "value": "..."}.
        """
        return await core.create_datasource(
            deps(), source_id, name, base_url, operations_json, description,
            kind, auth_json,
        )

    async def update_datasource(
        source_id: str,
        name: str | None = None,
        description: str | None = None,
        base_url: str | None = None,
        operations_json: str | None = None,
        auth_json: str | None = None,
    ) -> str:
        """Update a data source definition. Only provided fields change.

        Args:
            source_id: The data source id or name.
            name: New display name (omit to keep current).
            description: New description (omit to keep current).
            base_url: New base URL (omit to keep current).
            operations_json: JSON array replacing ALL operations (omit to keep current).
            auth_json: JSON auth block replacing the current one (omit to keep current).
        """
        return await core.update_datasource(
            deps(), source_id, name, description, base_url, operations_json, auth_json
        )

    async def delete_datasource(source_id: str) -> str:
        """Permanently delete a data source definition and unpublish its tools.

        Args:
            source_id: The data source id or name.
        """
        return await core.delete_datasource(deps(), source_id)

    async def import_datasource_schema(
        schema_url: str, kind: str = "http", auth_json: str = ""
    ) -> str:
        """List the operations an API specification defines. Stores nothing.

        Use this before creating a data source for an API that publishes a
        specification — it is the only way to get exact paths, params and
        response schemas without guessing.

        Args:
            schema_url: URL of an OpenAPI/Swagger document (JSON or YAML), a
                GraphQL introspection result, GraphQL SDL — or, with
                kind="graphql", the GraphQL endpoint itself (it is introspected).
            kind: "http" or "graphql".
            auth_json: Optional auth block if the schema URL needs credentials,
                same shape as create_datasource's auth_json.

        Returns:
            The declared base URL plus one line per operation. Pass the names you
            want to create_datasource_from_schema or
            add_datasource_operations_from_schema — the parsed operations are
            copied as-is, so never re-type them yourself.
        """
        return await core.import_datasource_schema(deps(), schema_url, kind, auth_json)

    async def create_datasource_from_schema(
        source_id: str,
        name: str,
        schema_url: str,
        operation_names: str,
        base_url: str = "",
        description: str = "",
        kind: str = "http",
        auth_json: str = "",
    ) -> str:
        """Create a data source from a specification, keeping only the named operations.

        Args:
            source_id: Unique kebab-case identifier (e.g. "github-api").
            name: Human-readable display name.
            schema_url: Same as import_datasource_schema.
            operation_names: Comma-separated operation names from
                import_datasource_schema's output.
            base_url: Base URL for the calls; defaults to the one the
                specification declares.
            description: What this data source provides.
            kind: "http" or "graphql".
            auth_json: Auth block for the API calls (also used to fetch the schema).
        """
        return await core.create_datasource_from_schema(
            deps(), source_id, name, schema_url, operation_names, base_url,
            description, kind, auth_json,
        )

    async def add_datasource_operations_from_schema(
        source_id: str,
        schema_url: str,
        operation_names: str,
        kind: str = "http",
        auth_json: str = "",
    ) -> str:
        """Add operations from a specification to an existing data source.

        Existing operations are kept; a name that already exists is skipped
        rather than overwritten.

        Args:
            source_id: The data source id or name.
            schema_url: Same as import_datasource_schema.
            operation_names: Comma-separated operation names to add.
            kind: "http" or "graphql".
            auth_json: Optional auth block if the schema URL needs credentials.
        """
        return await core.add_datasource_operations_from_schema(
            deps(), source_id, schema_url, operation_names, kind, auth_json
        )

    # ── run control ─────────────────────────────────────────────────────────

    async def terminate_run(run_id: str) -> str:
        """Stop a running run immediately and mark it as FAILED.

        Irreversible: any agent container attached to the run is killed and
        the run is recorded with error "Terminated by user". Only runs that
        are currently running or waiting on an agent can be terminated; use
        retry_run afterwards if the work should continue.

        Args:
            run_id: The run ID to terminate.
        """
        container = container_getter()
        try:
            run = await run_control.terminate_run(container, run_id)
        except RunControlError as exc:
            return f"Error ({exc.status_code}): {exc.detail}"
        return f"Run {run.id} terminated and marked failed."

    async def retry_run(run_id: str) -> str:
        """Retry a FAILED run from its last completed step.

        Already-finished steps are kept and skipped; the failed step and
        every step after it are reset to pending and executed again. Returns
        as soon as the retry is scheduled — poll get_run for progress.

        Args:
            run_id: The run ID to retry. The run must be in status failed.
        """
        container = container_getter()
        try:
            run, runner, resume_input = await run_control.retry_run(container, run_id)
        except RunControlError as exc:
            return f"Error ({exc.status_code}): {exc.detail}"
        _spawn_background(
            run_control._retry_graph(runner, run, container, resume_input),
            f"retry of run {run.id}",
        )
        return f"Run {run.id} retrying from the last completed step."

    async def restart_from_step(run_id: str, step_id: str) -> str:
        """Re-run a run from a specific step, discarding that step's results.

        Irreversible: the outputs and statuses of *step_id* and of every
        later step are deleted before execution restarts. Steps before it are
        preserved. Cannot be used on a currently running, cancelled or
        rejected run. Returns as soon as the restart is scheduled.

        Args:
            run_id: The run ID to restart.
            step_id: The step to restart from (see get_run for step IDs).
        """
        container = container_getter()
        try:
            run, runner, resume_input = await run_control.restart_from_step(
                container, run_id, step_id
            )
        except RunControlError as exc:
            return f"Error ({exc.status_code}): {exc.detail}"
        _spawn_background(
            run_control._retry_graph(runner, run, container, resume_input),
            f"restart of run {run.id} from step '{step_id}'",
        )
        return f"Run {run.id} restarting from step '{step_id}'."

    async def approve_run(run_id: str, corrections_json: str = "") -> str:
        """Approve a run that is paused at a human-approval step.

        The decision is final and the run continues immediately. It is recorded
        in the run's approval history as approver_source="mcp", approver
        "management-mcp" — this endpoint carries no per-user identity.

        Args:
            run_id: The run ID. The run must be waiting for approval.
            corrections_json: Optional JSON object of state corrections to apply
                along with the approval (omit for a plain approval).
        """
        corrections: dict | None = None
        if corrections_json:
            try:
                corrections = json.loads(corrections_json)
            except json.JSONDecodeError as exc:
                return f"Invalid corrections_json: {exc}"
            if not isinstance(corrections, dict):
                return "corrections_json must be a JSON object."
        container = container_getter()
        try:
            run, runner = await run_control.approve_run(container, run_id)
        except RunControlError as exc:
            return f"Error ({exc.status_code}): {exc.detail}"
        _spawn_background(
            run_control._resume_approved(
                runner, run, container, corrections,
                approver_name=APPROVER_NAME,
                approver_id=None,
                approver_source=APPROVER_SOURCE,
            ),
            f"approval resume of run {run.id}",
        )
        return f"Run {run.id} approved; resuming."

    async def reject_run(run_id: str, reason: str = "") -> str:
        """Reject a run that is paused at a human-approval step.

        The decision is final: the run does not continue past the gate and ends
        up rejected (or takes the gate's rejection route). Recorded in the run's
        approval history as approver_source="mcp", approver "management-mcp".

        Args:
            run_id: The run ID. The run must be waiting for approval.
            reason: Optional explanation stored with the rejection.
        """
        container = container_getter()
        try:
            run, runner = await run_control.reject_run(container, run_id)
        except RunControlError as exc:
            return f"Error ({exc.status_code}): {exc.detail}"
        _spawn_background(
            run_control._resume_rejected(
                runner, run, container, reason or None,
                approver_name=APPROVER_NAME,
                approver_id=None,
                approver_source=APPROVER_SOURCE,
            ),
            f"rejection resume of run {run.id}",
        )
        return f"Run {run.id} rejected."

    handlers = [
        list_workflows, run_workflow, list_runs, get_run,
        create_workflow, update_workflow, delete_workflow,
        list_agents, get_agent, create_agent, update_agent, delete_agent,
        list_datasources, create_datasource, update_datasource,
        delete_datasource, import_datasource_schema,
        create_datasource_from_schema, add_datasource_operations_from_schema,
        terminate_run, retry_run, restart_from_step, approve_run, reject_run,
    ]
    for handler in handlers:
        mcp.add_tool(handler, name=handler.__name__)
    logger.info("management MCP: registered %d tool(s)", len(handlers))
