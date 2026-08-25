"""Default chat agent — the entry point for all copilot chat interactions.

A ReAct-style LangGraph agent backed by the bundled workflow_assistant.yaml
config.  This is the platform's *internal* agent: it runs in-process and builds
the platform itself — workflows, agent definitions and data sources — rather
than delegating that to an external service.  Alongside the platform CRUD tools
it gets an ask_user tool that pauses execution via interrupt() to collect
clarifying answers, and the tools of every configured MCP server (optionally
narrowed by ``mcp_servers`` in the YAML config), which is what lets it act on
Jira/GitHub/Slack and on the ``datasources`` server's published operations.

The tool list is resolved per invocation, not frozen at build time: MCP servers
are (re)connected while the process runs — notably ``datasources``, whose tool
list changes whenever a data source is saved — and the agent must see the tools
that exist *now*.

The system prompt and LLM provider are loaded from the YAML config so the
agent can be customised without code changes.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Annotated, Any

from copilotkit import CopilotKitState
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig, interrupt

from app.application import management_tools as core
from app.application import run_control
from app.application.management_tools import ManagementDeps, _spawn_background
from app.application.run_control import RunControlError
from app.infrastructure.orchestration.yaml_graph import stream_graph_to_pause

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.tools import BaseTool

    from app.infrastructure.config.graph_loader import YamlGraphRegistry
    from app.infrastructure.persistence.agent_backend import AgentDefinitionBackend
    from app.infrastructure.persistence.data_source_backend import DataSourceDefinitionBackend
    from app.infrastructure.persistence.event_backend import EventDefinitionBackend
    from app.infrastructure.persistence.mongo import MongoGraphRunRepository
    from app.infrastructure.persistence.workflow_backend import WorkflowDefinitionBackend
    from app.infrastructure.tools.mcp_client import McpToolsProvider

logger = logging.getLogger(__name__)


_DEFAULT_SYSTEM_PROMPT = """\
You are the Workflow Assistant for this workflow automation platform.
Use your tools to help users run, inspect, and understand workflows.
"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AssistantState(CopilotKitState, total=False):  # type: ignore[misc]
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_default_workflow(
    llm: BaseChatModel,
    registry: "YamlGraphRegistry",
    run_repository: "MongoGraphRunRepository",
    checkpointer: BaseCheckpointSaver | None = None,
    agent_config: dict | None = None,
    workflow_backend: "WorkflowDefinitionBackend | None" = None,
    refresh_runner: "Callable[[str], Awaitable[None]] | None" = None,
    agent_backend: "AgentDefinitionBackend | None" = None,
    data_source_backend: "DataSourceDefinitionBackend | None" = None,
    event_backend: "EventDefinitionBackend | None" = None,
    refresh_datasources: "Callable[[], Awaitable[None]] | None" = None,
    mcp_tools_provider: "McpToolsProvider | None" = None,
    container: Any | None = None,
):
    """Build and compile the default ReAct chat agent.

    ``container`` is the ApplicationContainer.  It is optional: the run-control
    tools (terminate/retry/restart/approve/reject) need it and are only
    registered when it is passed, so callers that build the agent without a
    container (tests, embedded uses) keep the CRUD tool set unchanged.
    """

    config = agent_config or {}
    system_prompt_template = config.get("system_prompt", _DEFAULT_SYSTEM_PROMPT).strip()
    # Optional allow-list of MCP server names; None (the default) means every
    # configured server. An empty list means "no MCP tools at all".
    raw_servers = config.get("mcp_servers")
    allowed_mcp_servers: set[str] | None = (
        {str(s) for s in raw_servers} if isinstance(raw_servers, list) else None
    )

    # ── tools ────────────────────────────────────────────────────────────────

    # Every platform tool below is a thin wrapper over the shared cores in
    # app.application.management_tools — the same functions back the
    # /mcp/management MCP surface, so the two can never drift apart.  Only the
    # names, signatures and docstrings live here: those are the LLM-facing
    # contract of this agent.
    deps = ManagementDeps(
        registry=registry,
        run_repository=run_repository,
        workflow_backend=workflow_backend,
        agent_backend=agent_backend,
        data_source_backend=data_source_backend,
        event_backend=event_backend,
        refresh_runner=refresh_runner,
        refresh_datasources=refresh_datasources,
    )

    @tool
    def list_workflows() -> str:
        """List all available workflow IDs, names, and descriptions."""
        return core.list_workflows(deps)

    @tool
    async def run_workflow(workflow_id: str, request: str) -> str:
        """Start a workflow run.

        Args:
            workflow_id: The workflow ID (from list_workflows).
            request: A detailed description of the task to execute.

        Returns:
            JSON with run_id, workflow_id, workflow_name, and __event__ = workflow_started.
        """
        # stream_graph_to_pause is passed explicitly so that this module's
        # binding (the one tests patch) is the one the core calls.
        return await core.run_workflow(
            deps, workflow_id, request, stream_fn=stream_graph_to_pause
        )

    @tool
    async def list_runs(workflow_id: str | None = None, limit: int = 10) -> str:
        """List recent workflow runs.

        Args:
            workflow_id: Filter to a specific workflow (optional).
            limit: Maximum number of runs to return (default 10).
        """
        return await core.list_runs(deps, workflow_id, limit)

    @tool
    async def get_run(run_id: str) -> str:
        """Get detailed status and step-level output for a specific workflow run.

        Args:
            run_id: The run ID to inspect.
        """
        return await core.get_run(deps, run_id)

    @tool
    def ask_user(questions: list[str]) -> str:
        """Pause and ask the user clarifying questions before proceeding.

        Use this only when you genuinely cannot act without more information.
        Ask 1-3 focused questions.

        Args:
            questions: List of questions to ask the user.
        """
        answers: dict = interrupt({"type": "ask_context", "questions": questions})
        return "\n".join(
            f"Q: {q}\nA: {answers.get(str(i), '').strip()}"
            for i, q in enumerate(questions)
        )

    @tool
    async def get_workflow(workflow_id: str, include_steps: bool = True) -> str:
        """Read one workflow in full: its flags and its complete step list.

        Call this before update_workflow: that tool replaces the entire step
        list, so composing an update without the current steps can only
        overwrite them. Also the only way to see `use_storage`, which decides
        whether `storage` steps work at all.

        Args:
            workflow_id: The workflow id or name.
            include_steps: False for flags only, when the step list is long.
        """
        return await core.get_workflow(deps, workflow_id, include_steps)

    @tool
    async def create_workflow(
        workflow_id: str, name: str, description: str, steps_json: str,
        use_storage: bool = False, enabled: bool = True,
    ) -> str:
        """Create a new workflow definition and register it immediately.

        Args:
            workflow_id: Unique kebab-case identifier (e.g. "send-slack-report").
            name: Human-readable display name.
            description: What this workflow does.
            steps_json: JSON array of step objects. Each step must have "id" and "type".
                Supported types: http (webhook trigger), cron (scheduled trigger),
                llm (free-form LLM),
                mcp (single MCP tool call), human_approval (pause for approval),
                execute (OpenHands code execution), workflow (child workflow),
                http_call (outbound HTTP), langgraph-agent, claude-agent,
                data_source (invoke a DataSourceDefinition operation).
                Example: [{"id": "trigger", "type": "http"}, {"id": "classify", "type": "llm", "system_prompt": "...", "user_template": "...", "output_key": "verdict"}]
                Also available: storage (this workflow's own key/value state),
                slack (post/reply/read/DM/delete via a messaging provider).
            use_storage: Turn on this workflow's private key/value storage. Off by
                default, and a `storage` step in a workflow that has it off fails
                loudly rather than quietly not persisting -- so set it while
                creating, not afterwards.
            enabled: Pass False to create the workflow already disabled. A
                workflow carrying a `cron` or `pubsub` trigger is eligible to fire
                as soon as it is registered, so one that is not ready yet must be
                created disabled rather than disabled in a second call.
        """
        return await core.create_workflow(
            deps, workflow_id, name, description, steps_json,
            use_storage=use_storage, enabled=enabled,
        )

    @tool
    async def update_workflow(
        workflow_id: str,
        name: str | None = None,
        description: str | None = None,
        steps_json: str = "",
        enabled: bool | None = None,
        use_storage: bool | None = None,
    ) -> str:
        """Update an existing workflow definition (name, description, steps, enabled).

        Args:
            workflow_id: The workflow ID to update.
            name: New display name (omit to keep current).
            description: New description (omit to keep current).
            steps_json: JSON array replacing ALL steps (omit to keep current).
            enabled: False disables the workflow — every trigger and every manual
                start is refused until it is set back to True. Omit to keep current.
        """
        # Annotated `str`, not `str | None`, to stay identical to the MCP tool
        # of the same name (the surfaces are asserted to match): FastMCP
        # pre-parses a JSON string argument unless the annotation is exactly
        # `str`, which makes a `str | None` JSON field impossible to set.
        # Empty string means "omitted".
        return await core.update_workflow(
            deps, workflow_id, name, description, steps_json or None, enabled,
            use_storage,
        )

    @tool
    async def delete_workflow(workflow_id: str) -> str:
        """Permanently delete a workflow definition.

        Args:
            workflow_id: The workflow ID to delete.
        """
        return await core.delete_workflow(deps, workflow_id)

    # --- Agent tools ---

    @tool
    async def list_agents() -> str:
        """List all available agent definitions."""
        return await core.list_agents(deps)

    @tool
    async def get_agent(agent_id: str) -> str:
        """Get full agent definition by id or name."""
        return await core.get_agent(deps, agent_id)

    @tool
    async def create_agent(agent_id: str, name: str, description: str = "", default_runtime: str = "local", agent_input_json: str = "{}") -> str:
        """Create a new agent definition. agent_input_json is a JSON object of default input overrides."""
        return await core.create_agent(
            deps, agent_id, name, description, default_runtime, agent_input_json
        )

    @tool
    async def update_agent(agent_id: str, name: str = None, description: str = None, default_runtime: str = None, agent_input_json: str = None) -> str:
        """Update an existing agent definition. Only provided fields are changed; others preserved."""
        return await core.update_agent(
            deps, agent_id, name, description, default_runtime, agent_input_json
        )

    @tool
    async def delete_agent(agent_id: str) -> str:
        """Delete an agent definition by exact id."""
        return await core.delete_agent(deps, agent_id)

    # --- Data source tools ---

    @tool
    async def list_datasources() -> str:
        """List all registered data sources with their operations."""
        return await core.list_datasources(deps)

    @tool
    async def get_datasource(source_id: str) -> str:
        """Read one data source in full: auth type and every operation's shape.

        Read this before calling update_datasource: that tool replaces the whole
        operation list, so without the current definition an update silently
        drops the operations it does not mention. Secrets are never returned.

        Args:
            source_id: The data source id or name.
        """
        return await core.get_datasource(deps, source_id)

    @tool
    async def list_scripts() -> str:
        """List the Python scripts in the library that `python` steps reference."""
        return await core.list_scripts(deps)

    @tool
    async def get_script(script_id: str, include_code: bool = True) -> str:
        """Read one script from the library.

        Args:
            script_id: The script id or name.
            include_code: False for metadata only, when the body is long.
        """
        return await core.get_script(deps, script_id, include_code)

    @tool
    async def create_script(
        script_id: str, name: str, code: str, description: str = ""
    ) -> str:
        """Add a Python script to the library for `python` steps to reference.

        Args:
            script_id: Unique kebab-case identifier (e.g. "csm-deadline-compute").
            name: Human-readable display name, unique in the library.
            code: The script body. Runs with a `state` dict in scope and must set
                `output`. It cannot import project modules, so it must be
                self-contained; only the sandbox's allowed stdlib is available.
            description: What the script does.
        """
        return await core.create_script(deps, script_id, name, code, description)

    @tool
    async def update_script(
        script_id: str,
        name: str | None = None,
        code: str | None = None,
        description: str | None = None,
    ) -> str:
        """Change a library script; omitted fields keep their stored value.

        Args:
            script_id: The script id or name.
            name: New display name (omit to keep current).
            code: New body (omit to keep current).
            description: New description (omit to keep current).
        """
        return await core.update_script(deps, script_id, name, code, description)

    @tool
    async def delete_script(script_id: str) -> str:
        """Remove a script from the library.

        Refused while a workflow step still references it.

        Args:
            script_id: The script id or name.
        """
        return await core.delete_script(deps, script_id)

    @tool
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
            deps, source_id, name, base_url, operations_json, description, kind, auth_json
        )

    @tool
    async def update_datasource(
        source_id: str,
        name: str | None = None,
        description: str | None = None,
        base_url: str | None = None,
        operations_json: str = "",
        auth_json: str = "",
        pubsub_json: str = "",
    ) -> str:
        """Update a data source definition. Only provided fields change.

        Args:
            source_id: The data source id or name.
            name: New display name (omit to keep current).
            description: New description (omit to keep current).
            base_url: New base URL (omit to keep current).
            operations_json: JSON array replacing ALL operations (omit to keep current).
            auth_json: JSON auth block replacing the current one (omit to keep current).
            pubsub_json: For kind="pubsub" sources — JSON object
                {topic, subscription, project_id, event_schema} replacing the
                current Pub/Sub block (omit to keep current).
        """
        # Annotated `str`, not `str | None`, to stay identical to the MCP tool
        # of the same name (the surfaces are asserted to match): FastMCP
        # pre-parses a JSON string argument unless the annotation is exactly
        # `str`, which makes a `str | None` JSON field impossible to set.
        # Empty string means "omitted".
        return await core.update_datasource(
            deps, source_id, name, description, base_url,
            operations_json or None, auth_json or None, pubsub_json or None,
        )

    @tool
    async def create_pubsub_datasource(
        source_id: str,
        name: str,
        topic: str,
        event_schema_json: str = "",
        subscription: str = "",
        project_id: str = "",
        description: str = "",
    ) -> str:
        """Deprecated — use create_event. Registers the topic as an event.

        Args:
            source_id: Unique kebab-case identifier (e.g. "orders-events").
            name: Human-readable display name.
            topic: Topic short name ("orders") or full path
                ("projects/<project>/topics/orders").
            event_schema_json: Optional JSON object describing the message payload —
                same shape as an operation response_schema ({"type": "object",
                "required": [...], "properties": {...}}). Events that do not
                match it never start a run.
            subscription: Existing subscription to pull from. Leave empty to have
                one created on first use and saved back onto this event.
            project_id: Project override; empty uses the backend's PUBSUB_PROJECT_ID.
            description: What this topic carries.
        """
        return await core.create_pubsub_datasource(
            deps, source_id, name, topic, event_schema_json, subscription, project_id,
            description,
        )

    @tool
    async def list_events() -> str:
        """List the events (Pub/Sub topics) workflows can be triggered by."""
        return await core.list_events(deps)

    @tool
    async def get_event(event_id: str) -> str:
        """Read one event in full, including its payload schema.

        list_events omits the schema, so this is the only way to see what an
        event validates incoming messages against — and a message that fails
        that schema never starts a run.

        Args:
            event_id: The event id or name.
        """
        return await core.get_event(deps, event_id)

    @tool
    async def create_event(
        event_id: str,
        name: str,
        topic: str,
        event_schema_json: str = "",
        subscription: str = "",
        project_id: str = "",
        description: str = "",
    ) -> str:
        """Register a Pub/Sub topic as a reusable event for triggers.

        A `pubsub` trigger step can then say `event: <event_id>` instead of
        repeating the topic, event schema and subscription in every workflow.

        Args:
            event_id: Unique kebab-case identifier (e.g. "orders-events").
            name: Human-readable display name. Refused when another event
                already uses it — rename, or update that event instead.
            topic: Topic short name ("orders") or full path
                ("projects/<project>/topics/orders").
            event_schema_json: Optional JSON object describing the message payload —
                same shape as an operation response_schema ({"type": "object",
                "required": [...], "properties": {...}}). Events that do not
                match it never start a run.
            subscription: Existing subscription to pull from. Leave empty to have
                one created on first use and saved back onto this event.
            project_id: Project override; empty uses the backend's PUBSUB_PROJECT_ID.
            description: What this topic carries.
        """
        return await core.create_event(
            deps, event_id, name, topic, event_schema_json, subscription, project_id,
            description,
        )

    @tool
    async def update_event(
        event_id: str,
        name: str | None = None,
        description: str | None = None,
        topic: str | None = None,
        subscription: str | None = None,
        project_id: str | None = None,
        event_schema_json: str = "",
    ) -> str:
        """Change an existing event; omitted fields keep their stored value.

        Args:
            event_id: The event id or name.
            name: New display name (omit to keep current).
            description: New description (omit to keep current).
            topic: New topic (omit to keep current).
            subscription: New subscription to pull from (omit to keep current).
            project_id: New project override (omit to keep current).
            event_schema_json: JSON object replacing the event schema (omit to
                keep current).
        """
        # Annotated `str`, not `str | None`, to stay identical to the MCP tool
        # of the same name (the surfaces are asserted to match): FastMCP
        # pre-parses a JSON string argument unless the annotation is exactly
        # `str`, which makes a `str | None` JSON field impossible to set.
        # Empty string means "omitted".
        return await core.update_event(
            deps, event_id, name, description, topic, subscription, project_id,
            event_schema_json or None,
        )

    @tool
    async def delete_event(event_id: str) -> str:
        """Permanently delete an event.

        Workflow steps still pointing at it stop resolving, so check
        list_workflows first.

        Args:
            event_id: The event id or name.
        """
        return await core.delete_event(deps, event_id)

    @tool
    async def list_pubsub_subscriptions() -> str:
        """List which workflow steps are subscribed to Pub/Sub, and to what."""
        return core.list_pubsub_subscriptions(deps)

    # --- Data source schema import ---
    #
    # Specifications are parsed by code, never re-typed by the model: the tools
    # below hand back operation *names*, and the create/extend tools copy the
    # parsed operation objects verbatim. That keeps a 500-endpoint OpenAPI
    # document out of the context window and out of the failure modes of
    # transcribing JSON.

    @tool
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
        return await core.import_datasource_schema(deps, schema_url, kind, auth_json)

    @tool
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
            deps, source_id, name, schema_url, operation_names, base_url,
            description, kind, auth_json,
        )

    @tool
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
            deps, source_id, schema_url, operation_names, kind, auth_json
        )

    @tool
    async def delete_datasource(source_id: str) -> str:
        """Permanently delete a data source definition and unpublish its tools.

        Args:
            source_id: The data source id or name.
        """
        return await core.delete_datasource(deps, source_id)

    # ── messaging ───────────────────────────────────────────────────────────
    #
    # Same provider abstraction as the `slack` workflow step, so an operator
    # posting from here and a workflow step posting from a run share one
    # implementation.  No tool takes a credential: providers read theirs from
    # settings.

    @tool
    async def post_message(channel: str, text: str, thread_id: str = "", provider: str = "slack") -> str:
        """Post a message to a chat channel.

        Call read_messages first when you need the channel's recent context.
        The provider's credential comes from this deployment's settings, so
        there is no token to pass and none is accepted.

        Args:
            channel: Channel id to post into (e.g. a Slack channel id).
            text: Message body. The slack provider renders Slack mrkdwn.
            thread_id: Post as a reply inside this thread instead of as a new
                top-level message.
            provider: Messaging provider name (default "slack").
        """
        return await core.post_message(deps, channel, text, thread_id, provider)

    @tool
    async def read_messages(channel: str, limit: int = 20, oldest: str = "", provider: str = "slack") -> str:
        """Read a chat channel's recent messages, newest first.

        Args:
            channel: Channel id to read.
            limit: Maximum number of messages to return (default 20).
            oldest: Only messages at or after this provider timestamp.
            provider: Messaging provider name (default "slack").
        """
        return await core.read_messages(deps, channel, limit, oldest, provider)

    @tool
    async def read_thread(channel: str, thread_id: str, provider: str = "slack") -> str:
        """Read every message in one chat thread, the root message first.

        Use this before replying to check whether the reply is already there,
        so a re-run does not post it twice.

        Args:
            channel: Channel id the thread lives in.
            thread_id: Id of the thread's root message.
            provider: Messaging provider name (default "slack").
        """
        return await core.read_thread(deps, channel, thread_id, provider)

    @tool
    async def send_direct_message(user_id: str, text: str, provider: str = "slack") -> str:
        """Send a direct message to one user.

        Opens (or reuses) the DM channel and posts there, so nothing reaches a
        shared channel. This is the path for an alert that must not be
        broadcast — a false all-clear in a shared channel is worse than a DM
        nobody expected.

        Args:
            user_id: The provider's user id to send the DM to.
            text: Message body.
            provider: Messaging provider name (default "slack").
        """
        return await core.send_direct_message(deps, user_id, text, provider)

    @tool
    async def delete_message(channel: str, message_id: str, provider: str = "slack") -> str:
        """Permanently delete a message this platform posted.

        Irreversible, and only works for messages the platform's own bot
        created.

        Args:
            channel: Channel id the message is in.
            message_id: Id of the message to delete.
            provider: Messaging provider name (default "slack").
        """
        return await core.delete_message(deps, channel, message_id, provider)

    platform_tools = [list_workflows, get_workflow, run_workflow, list_runs, get_run, ask_user,
                      create_workflow, update_workflow, delete_workflow,
                      list_agents, get_agent, create_agent, update_agent, delete_agent,
                      list_datasources, get_datasource, create_datasource, update_datasource,
                      list_scripts, get_script, create_script, update_script, delete_script,
                      get_event,
                      create_pubsub_datasource, list_pubsub_subscriptions,
                      list_events, create_event, update_event, delete_event,
                      delete_datasource, import_datasource_schema,
                      create_datasource_from_schema,
                      add_datasource_operations_from_schema,
                      post_message, read_messages, read_thread,
                      send_direct_message, delete_message]

    # --- Run control tools ---
    #
    # These mutate live runs, so they need the ApplicationContainer (runners,
    # checkpoints, agent cleanup).  Same shared cores as the REST routes and
    # the /mcp/management surface; approvals made here are recorded with
    # approver_source="agent".
    if container is not None:

        @tool
        async def terminate_run(run_id: str) -> str:
            """Stop a running run immediately and mark it as FAILED.

            Irreversible: any agent container attached to the run is killed and
            the run is recorded with error "Terminated by user". Only runs that
            are currently running or waiting on an agent can be terminated; use
            retry_run afterwards if the work should continue.

            Args:
                run_id: The run ID to terminate.
            """
            try:
                run = await run_control.terminate_run(container, run_id)
            except RunControlError as exc:
                return f"Error ({exc.status_code}): {exc.detail}"
            return f"Run {run.id} terminated and marked failed."

        @tool
        async def retry_run(run_id: str) -> str:
            """Retry a FAILED run from its last completed step.

            Already-finished steps are kept and skipped; the failed step and
            every step after it are reset to pending and executed again. Returns
            as soon as the retry is scheduled — poll get_run for progress.

            Args:
                run_id: The run ID to retry. The run must be in status failed.
            """
            try:
                run, runner, resume_input = await run_control.retry_run(container, run_id)
            except RunControlError as exc:
                return f"Error ({exc.status_code}): {exc.detail}"
            _spawn_background(
                run_control._retry_graph(runner, run, container, resume_input),
                f"retry of run {run.id}",
            )
            return f"Run {run.id} retrying from the last completed step."

        @tool
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

        @tool
        async def approve_run(run_id: str, corrections_json: str = "") -> str:
            """Approve a run that is paused at a human-approval step.

            The decision is final and the run continues immediately; it is
            recorded in the run's approval history as approver_source="agent".
            Only approve when the user has actually asked for it.

            Args:
                run_id: The run ID. The run must be waiting for approval.
                corrections_json: Optional JSON object of state corrections to
                    apply along with the approval (omit for a plain approval).
            """
            corrections: dict | None = None
            if corrections_json:
                try:
                    corrections = json.loads(corrections_json)
                except json.JSONDecodeError as exc:
                    return f"Invalid corrections_json: {exc}"
                if not isinstance(corrections, dict):
                    return "corrections_json must be a JSON object."
            try:
                run, runner = await run_control.approve_run(container, run_id)
            except RunControlError as exc:
                return f"Error ({exc.status_code}): {exc.detail}"
            _spawn_background(
                run_control._resume_approved(
                    runner, run, container, corrections,
                    approver_name="chat-agent",
                    approver_id=None,
                    approver_source="agent",
                ),
                f"approval resume of run {run.id}",
            )
            return f"Run {run.id} approved; resuming."

        @tool
        async def reject_run(run_id: str, reason: str = "") -> str:
            """Reject a run that is paused at a human-approval step.

            The decision is final: the run does not continue past the gate and
            ends up rejected (or takes the gate's rejection route). Recorded in
            the run's approval history as approver_source="agent".

            Args:
                run_id: The run ID. The run must be waiting for approval.
                reason: Optional explanation stored with the rejection.
            """
            try:
                run, runner = await run_control.reject_run(container, run_id)
            except RunControlError as exc:
                return f"Error ({exc.status_code}): {exc.detail}"
            _spawn_background(
                run_control._resume_rejected(
                    runner, run, container, reason or None,
                    approver_name="chat-agent",
                    approver_id=None,
                    approver_source="agent",
                ),
                f"rejection resume of run {run.id}",
            )
            return f"Run {run.id} rejected."

        platform_tools.extend([
            terminate_run, retry_run, restart_from_step, approve_run, reject_run,
        ])

    _platform_tool_names = {t.name for t in platform_tools}

    # ── tool resolution ──────────────────────────────────────────────────────

    def _mcp_tools() -> "list[BaseTool]":
        """MCP tools available right now, narrowed by the config allow-list.

        Resolved per invocation because servers are (re)connected while the
        process runs: the ``datasources`` server republishes its tool list on
        every data source save, and a frozen list would keep serving the old one.
        """
        if mcp_tools_provider is None or allowed_mcp_servers == set():
            return []
        try:
            available = mcp_tools_provider.get_tools()
        except Exception:
            logger.warning("chat_agent: could not read MCP tools", exc_info=True)
            return []
        selected = []
        for mcp_tool in available:
            # A platform tool of the same name wins — the built-ins are the
            # documented contract of this agent.
            if mcp_tool.name in _platform_tool_names:
                continue
            if allowed_mcp_servers is not None:
                server = mcp_tools_provider.get_tool_server(mcp_tool.name)
                if server not in allowed_mcp_servers:
                    continue
            selected.append(mcp_tool)
        return selected

    def _all_tools() -> list:
        return [*platform_tools, *_mcp_tools()]

    # bind_tools() re-runs only when the tool set actually changed.
    _bound_llms: dict[tuple[str, ...], Any] = {}

    def _bound_llm(tools: list):
        key = tuple(t.name for t in tools)
        bound = _bound_llms.get(key)
        if bound is None:
            bound = llm.bind_tools(tools)
            _bound_llms[key] = bound
        return bound

    # ── nodes ────────────────────────────────────────────────────────────────

    def _build_system_prompt() -> str:
        """Build system prompt with current workflow list injected."""
        defs = registry.list_definitions()
        if defs:
            workflow_lines = "\n".join(
                f"- **{d['id']}**: {(d.get('description') or '').strip()}"
                for d in defs
            )
            return f"{system_prompt_template}\n\nAvailable workflows:\n{workflow_lines}"
        return system_prompt_template

    async def agent(state: AssistantState, config: RunnableConfig) -> dict:
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=_build_system_prompt())] + list(state.get("messages", []))
        response = await _bound_llm(_all_tools()).ainvoke(messages, config)
        return {"messages": [response]}

    async def tools_node(state: AssistantState, config: RunnableConfig) -> Any:
        """ToolNode over the current tool set (see ``_mcp_tools``)."""
        return await ToolNode(_all_tools()).ainvoke(state, config)

    def route(state: AssistantState) -> str:
        msgs = state.get("messages", [])
        if not msgs:
            return END
        last = msgs[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    # ── graph ────────────────────────────────────────────────────────────────

    sg: StateGraph = StateGraph(AssistantState)
    sg.add_node("agent", agent)
    sg.add_node("tools", tools_node)

    sg.add_edge(START, "agent")
    sg.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    sg.add_edge("tools", "agent")

    return sg.compile(checkpointer=checkpointer or MemorySaver())
