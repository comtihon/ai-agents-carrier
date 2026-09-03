from __future__ import annotations

import asyncio
import datetime
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from pymongo import MongoClient

from app.application.approval_service import ApprovalService
from app.application.run_control import WorkflowDisabledError, ensure_workflow_enabled
from app.core.config import Settings
from app.domain.models.event_definition import EventDefinition
from app.domain.models.graph_run import GraphRun
from app.infrastructure.config.graph_loader import (
    YamlGraphRegistry,
    build_registry_from_definitions,
    build_runner_from_definition,
)
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.orchestration.checkpointer import MongoDBCheckpointSaver
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner, stream_graph_to_pause
from app.infrastructure.persistence.mongo import MongoClientProvider, MongoGraphRunRepository, MongoPvcLeaseRepository, MongoAgentTaskRepository, MongoWarmPodRepository
from app.infrastructure.persistence.agent_backend import (
    AgentDefinitionBackend,
    MongoAgentBackend,
)
from app.infrastructure.persistence.data_source_backend import (
    DataSourceDefinitionBackend,
    MongoDataSourceBackend,
)
from app.infrastructure.persistence.event_backend import (
    EventDefinitionBackend,
    MongoEventBackend,
)
from app.infrastructure.persistence.script_backend import (
    MongoScriptBackend,
    ScriptDefinitionBackend,
)
from app.infrastructure.persistence.workflow_storage import (
    MongoWorkflowStorageBackend,
    WorkflowStorageBackend,
)
from app.infrastructure.persistence.workflow_backend import (
    LocalFilesWorkflowBackend,
    MongoWorkflowBackend,
    WorkflowDefinitionBackend,
)
from app.infrastructure.auth.service_token_provider import ServiceTokenProvider
from app.infrastructure.datasources.executor import DataSourceExecutor
from app.infrastructure.datasources.datastream import LocalDiskStreamStore, DataStreamStore
from app.infrastructure.persistence.data_artifact_backend import (
    DataArtifactBackend,
    MongoDataArtifactBackend,
)
from app.infrastructure.persistence.approval_backend import (
    ApprovalCaseBackend,
    MongoApprovalBackend,
)
from app.infrastructure.tools.mcp_client import McpToolsProvider
from app.infrastructure.triggers.cron_scheduler import CronScheduler
from app.infrastructure.triggers.pubsub_subscriber import (
    PubSubSubscriberManager,
    PubSubTriggerSpec,
)

logger = logging.getLogger(__name__)


@dataclass
class ApplicationContainer:
    settings: Settings
    llm: BaseChatModel
    mcp_tools_provider: McpToolsProvider
    yaml_graph_registry: YamlGraphRegistry
    mongo_provider: MongoClientProvider
    run_repository: MongoGraphRunRepository
    openhands: OpenHandsAdapter
    checkpointer: MongoDBCheckpointSaver | None = None
    # Workflow definition backend — None only in legacy test setups that
    # inject the registry directly (backward compat).
    workflow_backend: WorkflowDefinitionBackend | None = None
    # Agent definition backend — None when MongoDB is not configured or in
    # legacy test setups.  Required for langgraph-agent / claude-agent steps.
    agent_backend: AgentDefinitionBackend | None = None
    # Data source definition backend + its DAG executor — None when MongoDB is
    # not configured or in legacy test setups.  Required for `data_source`
    # steps and for the /mcp/datasources tools.
    data_source_backend: DataSourceDefinitionBackend | None = None
    data_source_executor: DataSourceExecutor | None = None
    # Reads back what the executor spilled; the `data_source`, `python` and
    # `llm` steps all go through it.
    stream_store: "DataStreamStore | None" = None
    # Run download manifests written by `data` steps and read by
    # GET /runs/{id}/data.  None in legacy test setups, which then run a `data`
    # step as a no-op that records nothing.
    data_artifact_backend: "DataArtifactBackend | None" = None
    # Privilege gate in front of destructive data-source operations: the store
    # of approval cases and the service that opens, decides and remembers them.
    # None in legacy test setups, which then run deletes ungated exactly as
    # before this feature existed.
    approval_backend: ApprovalCaseBackend | None = None
    approval_service: Any = None
    # Event definition backend — None when MongoDB is not configured or in
    # legacy test setups.  Required for `pubsub` trigger steps that name an
    # event, and for the subscription write-back.
    event_backend: EventDefinitionBackend | None = None
    # Python script library backend — None when MongoDB is not configured or in
    # legacy test setups.  Required for `python` steps that use `script_id`.
    script_backend: ScriptDefinitionBackend | None = None
    # Per-workflow key/value storage for `storage` steps. Shared instance; the
    # per-workflow scoping is enforced by passing the runner's own id on every
    # call, never by handing a runner a private backend.
    workflow_storage: WorkflowStorageBackend | None = None
    # Mints the service's own OAuth2 access token for outbound calls that use
    # `service_identity` auth — None in legacy test setups.
    service_token_provider: ServiceTokenProvider | None = None
    # Factory for per-step LLM overrides; None in legacy test setups.
    llm_factory: Callable[[str | None, str | None], BaseChatModel] | None = None
    # Runners keyed by run_id — alive for the duration of the run so that
    # approval-resume uses the exact definition snapshot from run start.
    live_runners: dict[str, YamlGraphRunner] = field(default_factory=dict)
    cron_scheduler: CronScheduler = field(default_factory=CronScheduler)
    # Built in startup() when PUBSUB_ENABLED — None means pubsub triggers are
    # parsed and stored but nothing subscribes to them.
    pubsub_subscriber: PubSubSubscriberManager | None = None
    pvc_lease_repository: MongoPvcLeaseRepository | None = None
    agent_task_repository: MongoAgentTaskRepository | None = None
    warm_pod_repository: MongoWarmPodRepository | None = None
    # References to background tasks created in startup() — held so they
    # can't be garbage-collected mid-flight, and cancelled on shutdown().
    _recover_task: asyncio.Task | None = field(default=None, init=False, repr=False, compare=False)
    _datasources_mcp_task: asyncio.Task | None = field(default=None, init=False, repr=False, compare=False)

    async def startup(self) -> None:
        # Fail fast on an enabled-but-incomplete outbound auth configuration.
        if self.service_token_provider is not None:
            self.service_token_provider.validate_configuration()
        # Cheap and idempotent, but it decides whether the run list can be
        # sorted at all: unindexed, Mongo sorts these large documents in
        # memory and the query aborts past 32 MB.
        try:
            await self.run_repository.ensure_indexes()
        except Exception:
            logger.exception("failed to ensure graph_runs indexes; run list may be slow")
        await self.mcp_tools_provider.start()
        self.cron_scheduler.start()
        if self.settings.pubsub_enabled and self.pubsub_subscriber is None:
            self.pubsub_subscriber = PubSubSubscriberManager(
                subscription_prefix=self.settings.pubsub_subscription_prefix,
                drop_invalid_messages=self.settings.pubsub_drop_invalid_messages,
                delete_orphaned_subscriptions=self.settings.pubsub_delete_orphaned_subscriptions,
                on_subscription_created=self._save_pubsub_subscription,
            )
        if self.pubsub_subscriber is not None:
            self.pubsub_subscriber.start()
        if self.workflow_backend is not None:
            await self._load_registry()
        # Confirm the store is usable before anything depends on it. A
        # misconfigured GCS store otherwise comes up Healthy and fails on the
        # first data source call of the deployment -- the worst place to learn
        # about it, and how the missing project surfaced.
        #
        # Off the event loop and bounded, both learned the hard way: this is a
        # blocking network call, and `startup()` runs inside the FastAPI
        # lifespan *before* uvicorn binds the port. Called inline it delayed
        # the bind, the liveness probe (initialDelaySeconds 10, period 20) got
        # connection refused, and the kubelet SIGKILLed the container before
        # it could serve -- a boot loop on a healthy cluster. So: a worker
        # thread, and a deadline after which a slow bucket check is a warning
        # rather than the reason a pod will not start.
        if self.stream_store is not None:
            check_ready = getattr(self.stream_store, "check_ready", None)
            if check_ready is not None:
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(check_ready),
                        timeout=self.settings.stream_ready_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    # Slow is not the same as broken. Refusing to boot here
                    # would make a sluggish metadata server an outage.
                    logger.warning(
                        "data stream store: readiness check did not finish "
                        "within %ss; continuing. If data source steps fail, "
                        "check the bucket and the service account's access.",
                        self.settings.stream_ready_timeout_seconds,
                    )
        # A local-disk stream does not survive the restart that just happened,
        # so anything already on disk belongs to a run that can no longer read
        # it. Sweeping at startup keeps a crash-loop from filling the node's
        # disk with results nothing will ever claim.
        if self.stream_store is not None:
            try:
                # Pinned streams are the exception: a `data` step told somebody
                # they could download those, so they are held to the much longer
                # artifact window instead of being swept with the backlog.
                await self.stream_store.purge_older_than(
                    self.settings.stream_ttl_seconds,
                    pinned_seconds=self.settings.data_artifact_ttl_seconds,
                )
            except Exception:
                logger.exception("failed to purge expired data streams")
        self._recover_task = asyncio.create_task(self._recover_incomplete_runs())
        # Data source MCP tools are loaded detached: the /mcp/datasources
        # endpoint only answers once uvicorn is serving, so this must never be
        # awaited inline during startup.
        if self.settings.mcp_datasources_enabled and self.data_source_backend is not None:
            self._datasources_mcp_task = asyncio.create_task(self._load_datasources_mcp())
        # Register 5-minute PVC lease cleanup sweeper
        if self.pvc_lease_repository is not None:
            from apscheduler.triggers.interval import IntervalTrigger
            from app.infrastructure.pvc_cleanup import cleanup_expired_pvcs
            _lease_repo = self.pvc_lease_repository
            _namespace = self.settings.agent_namespace
            async def _pvc_cleanup_job() -> None:
                await cleanup_expired_pvcs(_lease_repo, _namespace)
            self.cron_scheduler._scheduler.add_job(
                _pvc_cleanup_job,
                IntervalTrigger(minutes=5),
                id="pvc_lease_cleanup",
                replace_existing=True,
            )
        # Register 10-minute warm pod TTL cleanup sweeper
        if self.warm_pod_repository is not None:
            from apscheduler.triggers.interval import IntervalTrigger as _WIT
            self.cron_scheduler._scheduler.add_job(
                self._cleanup_expired_warm_pods,
                _WIT(minutes=10),
                id="warm_pod_ttl_cleanup",
                replace_existing=True,
            )
        # Register 60-second waiting_agent crash-detection sweeper
        from apscheduler.triggers.interval import IntervalTrigger as _IT
        self.cron_scheduler._scheduler.add_job(
            self._watch_waiting_agents,
            _IT(seconds=60),
            id="waiting_agent_watcher",
            replace_existing=True,
        )

    def _inject_runner_dependencies(self, runner: YamlGraphRunner) -> None:
        """Inject shared repositories onto a runner instance.

        Deliberately excludes `_callback_base_url` — its value is NOT uniform
        across call sites (some use `agent_callback_url or base_url`, one uses
        plain `base_url`, one doesn't set it at all), so each call site keeps
        setting that one itself to avoid silently changing existing behavior.
        """
        if self.agent_backend is not None:
            runner._agent_backend = self.agent_backend
        if self.pvc_lease_repository is not None:
            runner._pvc_lease_repository = self.pvc_lease_repository
        if self.agent_task_repository is not None:
            runner._agent_task_repository = self.agent_task_repository
        if self.warm_pod_repository is not None:
            runner._warm_pod_repository = self.warm_pod_repository
        if self.data_source_backend is not None:
            runner._data_source_backend = self.data_source_backend
        if self.data_source_executor is not None:
            runner._data_source_executor = self.data_source_executor
        if self.stream_store is not None:
            runner._stream_store = self.stream_store
        if self.data_artifact_backend is not None:
            runner._data_artifact_backend = self.data_artifact_backend
        if self.approval_service is not None:
            runner._approval_service = self.approval_service
        if self.script_backend is not None:
            runner._script_backend = self.script_backend
        if self.service_token_provider is not None:
            runner._service_token_provider = self.service_token_provider
        if self.workflow_storage is not None:
            runner._storage_backend = self.workflow_storage

    async def _load_datasources_mcp(self) -> None:
        """Publish data source MCP tools, then connect the local MCP server.

        Retried with backoff because the mounted /mcp/datasources endpoint is
        only reachable after the HTTP server has started accepting requests.
        """
        if self.data_source_backend is None:
            return
        from app.api.mcp.datasources_server import (
            get_datasources_mcp,
            rebuild_datasource_tools,
        )
        delay = 2.0
        for attempt in range(1, 7):
            await asyncio.sleep(delay)
            try:
                await rebuild_datasource_tools(
                    get_datasources_mcp(), self.data_source_backend, lambda: self
                )
                await self.mcp_tools_provider.refresh_server("datasources")
                logger.info("datasources MCP tools loaded")
                return
            except Exception:
                logger.debug(
                    "datasources MCP load attempt %d failed — retrying", attempt,
                    exc_info=True,
                )
                delay = min(delay * 2, 30.0)
        logger.warning("datasources MCP tools unavailable after 6 attempts")

    async def _load_registry(self) -> None:
        """Populate yaml_graph_registry from the configured backend."""
        assert self.workflow_backend is not None
        try:
            definitions = await self.workflow_backend.list()
        except Exception:
            logger.exception("Failed to load workflow definitions from backend — registry will be empty")
            return
        self.yaml_graph_registry = build_registry_from_definitions(
            definitions,
            llm=self.llm,
            llm_factory=self.llm_factory,
            mcp_tools_provider=self.mcp_tools_provider,
            openhands=self.openhands,
            run_repository=self.run_repository,
            checkpointer=self.checkpointer,
        )
        # Inject agent_backend and callback_base_url into every runner so that
        # agent steps can look up AgentDefinitions and build callback URLs.
        for wf_id in self.yaml_graph_registry.list_ids():
            runner = self.yaml_graph_registry.get(wf_id)
            if runner is not None:
                self._inject_runner_dependencies(runner)
                runner._callback_base_url = self.settings.agent_callback_url or self.settings.base_url
        logger.info("Loaded %d workflow definition(s) from backend", len(definitions))
        self._setup_all_cron_triggers()
        await self._setup_all_pubsub_triggers()

    def _setup_all_cron_triggers(self) -> None:
        for wf_id in self.yaml_graph_registry.list_ids():
            runner = self.yaml_graph_registry.get(wf_id)
            if runner:
                self._register_cron_steps(runner)

    def _register_cron_steps(self, runner: YamlGraphRunner) -> None:
        for step in runner.steps:
            if step.get("type") == "cron":
                schedule = step.get("schedule", "")
                if not schedule:
                    logger.warning(
                        "Cron step '%s' in workflow '%s' has no schedule — skipping",
                        step["id"], runner.id,
                    )
                    continue
                request_template = step.get("request_template", f"Scheduled run of {runner.id}")
                self.cron_scheduler.register(
                    runner.id,
                    step["id"],
                    schedule,
                    self._make_cron_job(runner.id, request_template),
                    timezone=str(step.get("timezone") or "UTC"),
                )

    # ── Pub/Sub triggers ─────────────────────────────────────────────────────

    async def _setup_all_pubsub_triggers(self) -> None:
        if self.pubsub_subscriber is None:
            return
        for wf_id in self.yaml_graph_registry.list_ids():
            runner = self.yaml_graph_registry.get(wf_id)
            if runner:
                await self._register_pubsub_steps(runner)

    async def _register_pubsub_steps(self, runner: YamlGraphRunner) -> None:
        """Make the subscriber's registrations match *runner*'s ``pubsub`` steps.

        Syncing rather than re-registering is what lets a removed ``pubsub`` node
        drop its subscription: steps the definition no longer has are released,
        and a subscription whose last consumer is gone is torn down.

        A step that cannot yield a spec (missing topic, unknown datasource) is
        logged and left out — one broken trigger must not stop the remaining
        workflows from loading.
        """
        if self.pubsub_subscriber is None:
            return
        entries = []
        for step in runner.steps:
            if step.get("type") != "pubsub":
                continue
            try:
                spec = await self._build_pubsub_spec(step)
            except Exception as exc:
                logger.error(
                    "Pub/Sub step '%s' in workflow '%s' is not usable: %s",
                    step["id"], runner.id, exc,
                )
                continue
            entries.append((step["id"], spec, self._make_pubsub_job(runner.id, step)))
        await self.pubsub_subscriber.sync_workflow(runner.id, entries)

    async def _build_pubsub_spec(self, step: dict) -> PubSubTriggerSpec:
        """Resolve a ``pubsub`` step into a subscriber spec.

        A step may name a pre-configured event, in which case that event
        supplies topic / subscription / schema and the step's own fields
        override whatever it sets.  ``datasource`` is the pre-events spelling
        of ``event`` and is still honoured: it resolves against the events
        first (the migration keeps ids), then against a leftover
        ``kind="pubsub"`` data source.
        """
        topic = (step.get("topic") or "").strip()
        subscription = (step.get("subscription") or "").strip()
        event_schema = step.get("schema") or None
        project_id = (step.get("project_id") or "").strip()
        event_id = (step.get("event") or step.get("datasource") or "").strip()

        if event_id:
            source = await self._resolve_event_source(event_id)
            topic = topic or source.topic
            subscription = subscription or source.subscription
            project_id = project_id or source.project_id
            event_schema = event_schema or source.event_schema

        if not topic:
            raise ValueError("no topic configured (set `topic`, or point `event` at an event)")

        return PubSubTriggerSpec(
            topic=topic,
            project_id=project_id or (self.settings.pubsub_project_id or ""),
            subscription=subscription,
            event_schema=event_schema,
            ack_deadline_seconds=int(step.get("ack_deadline_seconds") or self.settings.pubsub_ack_deadline_seconds),
            max_messages=int(step.get("max_messages") or self.settings.pubsub_max_messages),
            datasource_id=event_id,
        )

    async def _resolve_event_source(self, event_id: str) -> EventDefinition:
        """The event *event_id* names, from the events or from a legacy source.

        Raises ``ValueError`` when it resolves to nothing usable, so the caller
        can log the step as broken and leave the other triggers alone.
        """
        if self.event_backend is not None:
            event = await self.event_backend.get(event_id)
            if event is not None:
                return event
        if self.data_source_backend is None:
            if self.event_backend is None:
                raise ValueError(
                    f"step references event '{event_id}' but no event backend is configured"
                )
            raise ValueError(f"event '{event_id}' not found")
        source = await self.data_source_backend.get(event_id)
        if source is None:
            raise ValueError(f"event '{event_id}' not found")
        if source.kind != "pubsub" or source.pubsub is None:
            raise ValueError(f"'{event_id}' is not an event")
        return EventDefinition(
            id=source.id,
            name=source.name,
            description=source.description,
            topic=source.pubsub.topic,
            subscription=source.pubsub.subscription,
            project_id=source.pubsub.project_id,
            event_schema=source.pubsub.event_schema,
        )

    async def _save_pubsub_subscription(self, spec: PubSubTriggerSpec, subscription_path: str) -> None:
        """Persist a just-created subscription as an event.

        A subscription created from scratch is only useful to the next workflow
        if it can be found again, so it is written back: onto the event the
        step named, or as a new event keyed by topic.
        """
        if self.event_backend is None:
            return

        if spec.datasource_id:
            event = await self.event_backend.get(spec.datasource_id)
            if event is None:
                return
            if event.subscription == subscription_path:
                return
            event.subscription = subscription_path
            event.touch()
            await self.event_backend.update(event.id, event)
            logger.info(
                "Pub/Sub subscription %s saved onto event '%s'",
                subscription_path, spec.datasource_id,
            )
            return

        event_id = _pubsub_source_id(spec.topic)
        existing = await self.event_backend.get(event_id)
        if existing is not None:
            if existing.subscription != subscription_path:
                existing.subscription = subscription_path
                existing.touch()
                await self.event_backend.update(existing.id, existing)
            return

        defn = EventDefinition(
            id=event_id,
            name=spec.topic,
            description="Created from a workflow Pub/Sub trigger",
            topic=spec.topic,
            subscription=subscription_path,
            project_id=spec.project_id,
            event_schema=spec.event_schema,
        )
        defn.touch()
        await self.event_backend.create(defn)
        logger.info("Event '%s' created for topic %s", event_id, spec.topic)

    def _make_pubsub_job(self, workflow_id: str, step: dict):
        """Build the event callback that starts a run for one pubsub step."""
        step_id = step["id"]
        request_template = step.get("request_template", "")

        async def job(payload: dict, meta: dict) -> None:
            now = datetime.datetime.now(datetime.timezone.utc)
            request = request_template or json.dumps(payload)
            request = (
                request
                .replace("{now}", now.isoformat())
                .replace("{date}", now.strftime("%Y-%m-%d"))
            )
            try:
                if self.workflow_backend is not None:
                    defn = await self.workflow_backend.get(workflow_id)
                    if defn is None:
                        logger.warning("Pub/Sub job: workflow '%s' not found", workflow_id)
                        return
                    runner = build_runner_from_definition(
                        defn,
                        llm=self.llm,
                        llm_factory=self.llm_factory,
                        mcp_tools_provider=self.mcp_tools_provider,
                        registry=self.yaml_graph_registry,
                        run_repository=self.run_repository,
                        openhands=self.openhands,
                        checkpointer=self.checkpointer,
                    )
                    self._inject_runner_dependencies(runner)
                    runner._callback_base_url = self.settings.agent_callback_url or self.settings.base_url
                    definition_snapshot: dict | None = defn.to_raw_dict()
                else:
                    runner = self.yaml_graph_registry.get(workflow_id)
                    if runner is None:
                        logger.warning("Pub/Sub job: workflow '%s' not in registry", workflow_id)
                        return
                    definition_snapshot = None

                # A disabled workflow fires no triggers. The job stays registered
                # so re-enabling needs no restart; it just declines to run.
                try:
                    ensure_workflow_enabled(runner)
                except WorkflowDisabledError as exc:
                    logger.info("Pub/Sub trigger skipped: %s", exc.detail)
                    return

                thread_id = str(uuid4())
                self.live_runners[thread_id] = runner

                run = GraphRun(
                    id=thread_id,
                    graph_id=workflow_id,
                    user_request=request,
                    status="running",
                    workflow_definition=definition_snapshot,
                )
                await self.run_repository.create(run)
                run.step_statuses = {s["id"]: "pending" for s in runner.steps}

                trigger_info = {**meta, "triggered_at": now.isoformat(), "step_id": step_id}
                initial_state = {
                    "request": request,
                    "trigger_payload": payload,
                    "trigger_info": trigger_info,
                }
                logger.info(
                    "Pub/Sub trigger started run %s for workflow '%s' (message %s)",
                    thread_id, workflow_id, meta.get("message_id", "?"),
                )
                await stream_graph_to_pause(
                    runner, run, self.run_repository, initial_state, base_url=self.settings.base_url,
                )

                if run.status in ("completed", "failed", "cancelled", "rejected"):
                    self.live_runners.pop(thread_id, None)
            except Exception:
                logger.exception("Pub/Sub job execution failed for workflow '%s'", workflow_id)

        return job

    def _make_cron_job(self, workflow_id: str, request_template: str):
        async def job() -> None:
            now = datetime.datetime.now(datetime.timezone.utc)
            request = (
                request_template
                .replace("{now}", now.isoformat())
                .replace("{date}", now.strftime("%Y-%m-%d"))
            )
            try:
                if self.workflow_backend is not None:
                    defn = await self.workflow_backend.get(workflow_id)
                    if defn is None:
                        logger.warning("Cron job: workflow '%s' not found", workflow_id)
                        return
                    runner = build_runner_from_definition(
                        defn,
                        llm=self.llm,
                        llm_factory=self.llm_factory,
                        mcp_tools_provider=self.mcp_tools_provider,
                        registry=self.yaml_graph_registry,
                        run_repository=self.run_repository,
                        openhands=self.openhands,
                        checkpointer=self.checkpointer,
                    )
                    self._inject_runner_dependencies(runner)
                    definition_snapshot: dict | None = defn.to_raw_dict()
                else:
                    runner = self.yaml_graph_registry.get(workflow_id)
                    if runner is None:
                        logger.warning("Cron job: workflow '%s' not found in registry", workflow_id)
                        return
                    definition_snapshot = None

                # A disabled workflow fires no triggers. The job stays registered
                # so re-enabling needs no restart; it just declines to run.
                try:
                    ensure_workflow_enabled(runner)
                except WorkflowDisabledError as exc:
                    logger.info("Cron trigger skipped: %s", exc.detail)
                    return

                thread_id = str(uuid4())
                self.live_runners[thread_id] = runner

                run = GraphRun(
                    id=thread_id,
                    graph_id=workflow_id,
                    user_request=request,
                    status="running",
                    workflow_definition=definition_snapshot,
                )
                await self.run_repository.create(run)
                run.step_statuses = {s["id"]: "pending" for s in runner.steps}

                trigger_info = {
                    "triggered_at": now.isoformat(),
                    "type": "cron",
                }
                initial_state = {"request": request, "trigger_info": trigger_info}
                await stream_graph_to_pause(runner, run, self.run_repository, initial_state, base_url=self.settings.base_url)

                if run.status in ("completed", "failed", "cancelled", "rejected"):
                    self.live_runners.pop(thread_id, None)

            except Exception:
                logger.exception("Cron job execution failed for workflow '%s'", workflow_id)

        return job

    async def refresh_runner(self, workflow_id: str) -> None:
        """Rebuild the registry runner for *workflow_id* after a definition change.

        Existing live runners (in-flight runs) are NOT replaced — they keep the
        definition snapshot they started with.
        """
        if self.workflow_backend is None:
            return
        # Always clear stale cron jobs for this workflow first. Pub/Sub
        # registrations are *not* cleared here: dropping them before re-reading
        # the definition would cancel (and delete) a subscription the saved
        # definition still wants. _register_pubsub_steps syncs instead, and the
        # deleted-workflow branch below unregisters explicitly.
        self.cron_scheduler.unregister_workflow(workflow_id)
        defn = await self.workflow_backend.get(workflow_id)
        if defn is not None:
            runner = build_runner_from_definition(
                defn,
                llm=self.llm,
                llm_factory=self.llm_factory,
                mcp_tools_provider=self.mcp_tools_provider,
                registry=self.yaml_graph_registry,
                run_repository=self.run_repository,
                openhands=self.openhands,
                checkpointer=self.checkpointer,
            )
            self._inject_runner_dependencies(runner)
            runner._callback_base_url = self.settings.agent_callback_url or self.settings.base_url
            self.yaml_graph_registry._runners[workflow_id] = runner
            self._register_cron_steps(runner)
            await self._register_pubsub_steps(runner)
            logger.info("Registry runner refreshed for workflow '%s'", workflow_id)
        else:
            if self.pubsub_subscriber is not None:
                self.pubsub_subscriber.unregister_workflow(workflow_id)
            self.yaml_graph_registry._runners.pop(workflow_id, None)
            logger.info("Registry runner removed for workflow '%s'", workflow_id)

    async def _recover_incomplete_runs(self) -> None:
        """On startup, find runs still marked running/waiting_approval and restore them."""
        try:
            incomplete = await self.run_repository.list_incomplete()
        except Exception:
            logger.exception("Failed to query incomplete runs for recovery")
            return
        if not incomplete:
            return
        logger.info("Recovering %d incomplete run(s) after restart", len(incomplete))
        for run in incomplete:
            try:
                await self._recover_run(run)
            except Exception:
                logger.exception("Failed to recover run %s", run.id)

    async def _recover_run(self, run: GraphRun) -> None:
        runner = self._build_runner_for_recovery(run)
        if runner is None:
            logger.warning("run %s: cannot recover — workflow '%s' not available", run.id, run.graph_id)
            run.status = "failed"
            run.state = {"error": "Workflow definition not available after server restart"}
            run.touch()
            await self.run_repository.update(run)
            return

        # Reconstruct accumulated state from persisted step outputs (in step order)
        accumulated: dict[str, Any] = {"request": run.user_request}
        last_done: str | None = None
        for step in runner.steps:
            sid = step["id"]
            if run.step_statuses.get(sid) in ("finished", "skipped"):
                last_done = sid
                output = run.step_outputs.get(sid)
                if output and isinstance(output, dict):
                    accumulated.update(output)

        # Seed internal state keys (e.g. _openhands_conv_*, _conv_map) persisted
        # mid-step by _save_conv_id — these exist in run.state but not yet in
        # step_outputs when the server crashed before the step completed.
        if run.state and isinstance(run.state, dict):
            for k, v in run.state.items():
                if k.startswith("_") and v is not None:
                    accumulated.setdefault(k, v)

        config = {"configurable": {"thread_id": run.id}}

        if run.status == "waiting_agent":
            # If the agent URL is known, probe it — the pod may still be running.
            agent_url = run.agent_url
            agent_alive = False
            if agent_url:
                try:
                    from app.runtime.k8s import K8sRuntime
                    agent_alive = await K8sRuntime(namespace=self.settings.agent_namespace).is_alive(agent_url)
                except Exception:
                    pass
                if not agent_alive:
                    try:
                        from app.runtime.docker import DockerRuntime
                        agent_alive = await DockerRuntime(
                            registry_username=self.settings.docker_registry_username,
                            registry_password=self.settings.docker_registry_password,
                        ).is_alive(agent_url)
                    except Exception:
                        pass

            if agent_alive:
                # Pod survived the restart — reconnect by restoring the runner so
                # agent callbacks (/agent/output, /agent/question, etc.) can reach it.
                self.live_runners[run.id] = runner
                logger.info(
                    "run %s: waiting_agent on restart — agent at %s still alive, reconnected",
                    run.id, agent_url,
                )
                return

            # Agent is gone — clean up and mark failed.
            try:
                from app.runtime.k8s import K8sRuntime
                await K8sRuntime(namespace=self.settings.agent_namespace).terminate_by_run_id(None, run.id)
            except Exception:
                logger.debug("run %s: k8s release cleanup on recovery failed", run.id, exc_info=True)
            try:
                from app.runtime.docker import DockerRuntime
                await DockerRuntime(
                    registry_username=self.settings.docker_registry_username,
                    registry_password=self.settings.docker_registry_password,
                ).terminate_by_run_id(None, run.id)
            except Exception:
                logger.debug("run %s: docker cleanup on recovery failed", run.id, exc_info=True)
            run.status = "failed"
            run.agent_url = None
            run.state = {**(run.state or {}), "error": "Agent container lost due to server restart"}
            run.touch()
            await self.run_repository.update(run)
            logger.info("run %s: waiting_agent on restart — agent gone, marked failed", run.id)
            return

        if run.status == "waiting_approval":
            # With MongoDB checkpointer the interrupt state is already persisted.
            # Check for a valid checkpoint before falling back to re-execution.
            try:
                snap = await runner.graph.aget_state(config)
                has_checkpoint = bool(snap.next) or bool(getattr(snap, "interrupts", ()))
            except Exception:
                has_checkpoint = False

            if has_checkpoint:
                self.live_runners[run.id] = runner
                logger.info("run %s: waiting_approval restored from MongoDB checkpoint", run.id)
                return

            # No checkpoint (pre-MongoDB run) — fall back to re-execution
            if last_done is not None:
                try:
                    await runner.graph.aupdate_state(config, accumulated, as_node=last_done)
                except Exception:
                    logger.exception("run %s: aupdate_state failed during recovery", run.id)
                    run.status = "failed"
                    run.state = {"error": "State recovery failed after server restart"}
                    run.touch()
                    await self.run_repository.update(run)
                    return

            resume_input: Any = None if last_done else accumulated
            try:
                async for _ in runner.graph.astream(resume_input, config, stream_mode="updates"):
                    pass
            except Exception:
                logger.exception("run %s: approval interrupt refire failed", run.id)
                run.status = "failed"
                run.state = {"error": "Approval state recovery failed after server restart"}
                run.touch()
                await self.run_repository.update(run)
                return
            self.live_runners[run.id] = runner
            logger.info("run %s: waiting_approval re-armed (approval_step=%s)", run.id, run.current_step)

        else:  # "running"
            if last_done is not None:
                try:
                    await runner.graph.aupdate_state(config, accumulated, as_node=last_done)
                except Exception:
                    logger.exception("run %s: aupdate_state failed during recovery", run.id)
                    run.status = "failed"
                    run.state = {"error": "State recovery failed after server restart"}
                    run.touch()
                    await self.run_repository.update(run)
                    return

            resume_input = None if last_done else accumulated
            self.live_runners[run.id] = runner
            asyncio.create_task(self._resume_run(runner, run, resume_input))
            logger.info("run %s: resuming execution from last completed step=%s", run.id, last_done)

    async def _resume_run(self, runner: YamlGraphRunner, run: GraphRun, input_value: Any) -> None:
        try:
            await stream_graph_to_pause(runner, run, self.run_repository, input_value)
        except Exception:
            logger.exception("run %s: resumed execution failed", run.id)
        finally:
            if run.status in ("completed", "failed", "cancelled", "rejected"):
                self.live_runners.pop(run.id, None)

    def _build_runner_for_recovery(self, run: GraphRun) -> YamlGraphRunner | None:
        if run.workflow_definition is not None:
            # Build from the exact definition snapshot stored at run-start time
            try:
                runner = YamlGraphRunner(
                    run.workflow_definition,
                    llm=self.llm,
                    llm_factory=self.llm_factory,
                    mcp_tools_provider=self.mcp_tools_provider,
                    openhands=self.openhands,
                    checkpointer=self.checkpointer,
                )
                runner._registry = self.yaml_graph_registry
                runner._run_repository = self.run_repository
                self._inject_runner_dependencies(runner)
                runner._callback_base_url = self.settings.agent_callback_url or self.settings.base_url
                return runner
            except Exception:
                logger.exception("run %s: failed to build runner from definition snapshot", run.id)
        # Fall back to the live registry (e.g. legacy runs without a snapshot)
        return self.yaml_graph_registry.get(run.graph_id)

    async def _fail_lost_agent_run(self, run: GraphRun, reason: str) -> None:
        """Mark a waiting_agent run as failed and best-effort terminate its pod."""
        run.status = "failed"
        run.agent_url = None
        run.state = {**(run.state or {}), "error": reason}
        if run.current_step:
            run.step_statuses = {**(run.step_statuses or {}), run.current_step: "failed"}
            run.step_outputs = {**(run.step_outputs or {}), run.current_step: {"error": reason}}
        run.touch()
        await self.run_repository.update(run)
        try:
            from app.runtime.k8s import K8sRuntime
            await K8sRuntime(namespace=self.settings.agent_namespace).terminate_by_run_id(None, run.id)
        except Exception:
            logger.debug("run %s: k8s cleanup in watcher failed", run.id, exc_info=True)
        try:
            from app.runtime.docker import DockerRuntime
            await DockerRuntime(
                registry_username=self.settings.docker_registry_username,
                registry_password=self.settings.docker_registry_password,
            ).terminate_by_run_id(None, run.id)
        except Exception:
            logger.debug("run %s: docker cleanup in watcher failed", run.id, exc_info=True)

    async def _cleanup_expired_warm_pods(self) -> None:
        """Periodic sweep: helm-uninstall warm pods whose TTL has expired."""
        if self.warm_pod_repository is None:
            return
        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            expired = await self.warm_pod_repository.list_expired(now)
            for record in expired:
                try:
                    from app.runtime.k8s import K8sRuntime
                    rt = K8sRuntime(namespace=self.settings.agent_namespace)
                    await rt.uninstall_release(record.release_name)
                except Exception:
                    logger.debug("warm pod cleanup: uninstall '%s' failed", record.release_name, exc_info=True)
                try:
                    await self.warm_pod_repository.delete(record.run_id, record.agent_id)
                except Exception:
                    logger.debug("warm pod cleanup: delete record failed for %s|%s", record.run_id, record.agent_id, exc_info=True)
        except Exception:
            logger.exception("_cleanup_expired_warm_pods sweep failed")

    async def _watch_waiting_agents(self) -> None:
        """Periodic sweep: detect crashed agent pods for waiting_agent runs."""
        try:
            runs = await self.run_repository.list_incomplete()
            candidates = [r for r in runs if r.status == "waiting_agent" and r.agent_url]
            for run in candidates:
                try:
                    import httpx
                    alive = False
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as http:
                            resp = await http.get(f"{run.agent_url}/health")
                            alive = resp.is_success
                    except Exception:
                        alive = False
                    if not alive:
                        # TOCTOU guard: re-fetch before marking failed
                        fresh = await self.run_repository.get(run.id)
                        if fresh is not None and fresh.status == "waiting_agent":
                            logger.info(
                                "run %s: agent at %s unreachable — marking failed",
                                run.id, run.agent_url,
                            )
                            await self._fail_lost_agent_run(fresh, "Agent pod health check failed")
                except Exception:
                    logger.exception("run %s: error in waiting_agent watcher sweep", run.id)
        except Exception:
            logger.exception("_watch_waiting_agents sweep failed")

    async def shutdown(self) -> None:
        self.cron_scheduler.stop()
        if self.pubsub_subscriber is not None:
            self.pubsub_subscriber.stop()
        for task in (self._recover_task, self._datasources_mcp_task):
            if task is not None and not task.done():
                task.cancel()
        await self.mcp_tools_provider.stop()
        await self.mongo_provider.close()
        if self.checkpointer is not None:
            self.checkpointer.close()
        if isinstance(self.workflow_backend, MongoWorkflowBackend):
            await self.workflow_backend.close()
        if isinstance(self.agent_backend, MongoAgentBackend):
            await self.agent_backend.close()
        if isinstance(self.data_source_backend, MongoDataSourceBackend):
            await self.data_source_backend.close()
        if isinstance(self.approval_backend, MongoApprovalBackend):
            await self.approval_backend.close()
        if isinstance(self.event_backend, MongoEventBackend):
            await self.event_backend.close()


def _pubsub_source_id(topic: str) -> str:
    """Stable event id for a topic: ``pubsub-<last path segment>``."""
    name = topic.rsplit("/", 1)[-1] if topic else "topic"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower() or "topic"
    return f"pubsub-{slug}"


def _fake_llm(reason: str) -> BaseChatModel:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    return FakeMessagesListChatModel(responses=[AIMessage(content=reason)])


def build_llm_for(provider: str | None, model: str | None, settings: Settings) -> BaseChatModel:
    """Build an LLM for an integration name + optional model override.

    `provider` is the `name` of an entry in `LLM_INTEGRATIONS`. All integrations
    are treated as OpenAI/LiteLLM-compatible: a single ChatOpenAI client is
    constructed with the integration's `base_url`, `api_key`, and `model`.

    Resolution order for the model: step override > integration's `default_model`.
    """
    if not provider:
        return _fake_llm("LLM not configured. Set LLM_PROVIDER to one of the configured LLM_INTEGRATIONS.")
    integration = settings.get_llm_integration(provider)
    if integration is None:
        return _fake_llm(
            f"LLM integration '{provider}' is not defined in LLM_INTEGRATIONS."
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model or integration.default_model,
        api_key=integration.resolved_api_key(),  # type: ignore[arg-type]
        base_url=integration.base_url,
        max_tokens=16000,
    )


def build_llm(settings: Settings) -> BaseChatModel:
    return build_llm_native(settings.llm_provider, None, settings)


def build_llm_native(
    provider: str | None,
    model: str | None,
    settings: Settings,
    max_tokens: int = 8096,
) -> BaseChatModel:
    """Build an LLM supporting both LLM_INTEGRATIONS and standalone API keys.

    Resolution order:
    1. LLM_INTEGRATIONS lookup (OpenAI-compatible endpoint) — if a matching
       integration is configured, use it.
    2. Native provider via standalone API key fields on Settings.
       Supported: ``anthropic``, ``openai``, ``google``.
    3. Falls back to a fake LLM with an informative error message.
    """
    resolved_provider = provider or settings.llm_provider

    # 1. Try LLM_INTEGRATIONS first
    if resolved_provider and settings.get_llm_integration(resolved_provider):
        return build_llm_for(resolved_provider, model, settings)

    # 2. Native providers via standalone Settings fields
    if resolved_provider == "anthropic" or (not resolved_provider and settings.anthropic_api_key):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-opus-4-7",
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )
    if resolved_provider == "openai" or (not resolved_provider and settings.openai_api_key):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o",
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )
    if resolved_provider == "google" or (not resolved_provider and settings.google_api_key):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash",
            google_api_key=settings.google_api_key,
            max_output_tokens=max_tokens,
        )

    return _fake_llm(
        f"No LLM configured for provider '{resolved_provider}'. "
        "Set LLM_PROVIDER and configure either LLM_INTEGRATIONS or a standalone API key."
    )


def _make_llm_factory(settings: Settings) -> Callable[[str | None, str | None], BaseChatModel]:
    """Return a factory that builds an LLM with optional per-step provider/model overrides."""
    def factory(provider: str | None, model: str | None) -> BaseChatModel:
        return build_llm_for(provider or settings.llm_provider, model, settings)
    return factory


def _build_workflow_backend(settings: Settings) -> WorkflowDefinitionBackend:
    if settings.workflow_backend_type == "mongodb":
        return MongoWorkflowBackend(settings.mongodb_uri, settings.mongodb_database)
    # Local-files backend: treat every loaded definition as read-only because
    # in production the directory is a k8s ConfigMap volume (readOnly: true).
    return LocalFilesWorkflowBackend(settings.graph_definitions_path, readonly=False)


def _build_stream_store(settings: Settings) -> DataStreamStore:
    """The store every data source result is written to.

    Object storage is a second implementation of the same ABC, not a second
    kind of storage bolted alongside it, so selecting it changes nothing about
    how a result is produced or read — only where the bytes end up.  ``local``
    stays the default: an unconfigured deployment must keep behaving exactly as
    it did before this choice existed.
    """
    backend = (settings.stream_backend or "local").strip().lower()
    if backend == "gcs":
        from app.infrastructure.datasources.datastream_gcs import GcsStreamStore

        logger.info(
            "data stream store: gcs bucket '%s'%s%s",
            settings.stream_gcs_bucket,
            f" prefix '{settings.stream_gcs_prefix}'" if settings.stream_gcs_prefix else "",
            f" project '{settings.stream_gcs_project}'" if settings.stream_gcs_project else "",
        )
        return GcsStreamStore(
            settings.stream_gcs_bucket,
            prefix=settings.stream_gcs_prefix,
            project=settings.stream_gcs_project,
        )
    if backend != "local":
        # Refused rather than silently defaulted: a typo in STREAM_BACKEND
        # would otherwise put a deployment that meant to be durable back on
        # ephemeral pod disk, and nothing would say so until a restart.
        raise ValueError(
            f"STREAM_BACKEND '{settings.stream_backend}' is not a known data "
            f"stream store; use 'local' or 'gcs'"
        )
    return LocalDiskStreamStore(settings.stream_dir)


def build_container(settings: Settings) -> ApplicationContainer:
    llm = build_llm(settings)
    llm_factory = _make_llm_factory(settings)
    mcp_tools_provider = McpToolsProvider(settings)
    openhands = OpenHandsAdapter(settings)
    mongo_provider = MongoClientProvider(settings)
    run_repository = mongo_provider.get_repository()
    pvc_lease_repository = mongo_provider.get_pvc_lease_repository()
    agent_task_repository = mongo_provider.get_agent_task_repository()
    warm_pod_repository = mongo_provider.get_warm_pod_repository()
    workflow_backend = _build_workflow_backend(settings)
    # Agent definitions are always stored in MongoDB (no local-files backend).
    agent_backend = MongoAgentBackend(settings.mongodb_uri, settings.mongodb_database)
    # Data source definitions are likewise MongoDB-only.
    data_source_backend = MongoDataSourceBackend(settings.mongodb_uri, settings.mongodb_database)
    # Events (Pub/Sub topics workflows subscribe to) are likewise MongoDB-only.
    event_backend = MongoEventBackend(settings.mongodb_uri, settings.mongodb_database)
    # Python script library is likewise MongoDB-only.
    script_backend = MongoScriptBackend(settings.mongodb_uri, settings.mongodb_database)
    # Per-workflow storage: one collection, entries owned by workflow id.
    workflow_storage = MongoWorkflowStorageBackend(settings.mongodb_uri, settings.mongodb_database)
    service_token_provider = ServiceTokenProvider(settings)
    # Where every data source result is written.
    # Required, not optional: results do not travel as values, so every
    # data source call needs somewhere to write its stream.
    stream_store = _build_stream_store(settings)
    data_source_executor = DataSourceExecutor(
        token_provider=service_token_provider, stream_store=stream_store
    )
    approval_backend = MongoApprovalBackend(settings.mongodb_uri, settings.mongodb_database)
    approval_service = ApprovalService(
        approval_backend,
        settings,
        workflow_backend=workflow_backend,
        run_repository=run_repository,
    )
    checkpointer = MongoDBCheckpointSaver(
        MongoClient(settings.mongodb_uri),
        db_name=settings.mongodb_database,
    )
    # Registry starts empty; populated asynchronously in startup().
    return ApplicationContainer(
        settings=settings,
        llm=llm,
        llm_factory=llm_factory,
        mcp_tools_provider=mcp_tools_provider,
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=run_repository,
        openhands=openhands,
        workflow_backend=workflow_backend,
        agent_backend=agent_backend,
        data_source_backend=data_source_backend,
        data_source_executor=data_source_executor,
        stream_store=stream_store,
        data_artifact_backend=MongoDataArtifactBackend(
            settings.mongodb_uri, settings.mongodb_database
        ),
        approval_backend=approval_backend,
        approval_service=approval_service,
        event_backend=event_backend,
        script_backend=script_backend,
        workflow_storage=workflow_storage,
        service_token_provider=service_token_provider,
        checkpointer=checkpointer,
        pvc_lease_repository=pvc_lease_repository,
        agent_task_repository=agent_task_repository,
        warm_pod_repository=warm_pod_repository,
    )
