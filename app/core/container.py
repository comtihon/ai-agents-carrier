from __future__ import annotations

import asyncio
import datetime
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from pymongo import MongoClient

from app.core.config import Settings
from app.domain.models.graph_run import GraphRun
from app.infrastructure.config.graph_loader import (
    YamlGraphRegistry,
    build_registry_from_definitions,
    build_runner_from_definition,
)
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.orchestration.checkpointer import MongoDBCheckpointSaver
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner, stream_graph_to_pause
from app.infrastructure.persistence.mongo import MongoClientProvider, MongoGraphRunRepository, MongoPvcLeaseRepository
from app.infrastructure.persistence.agent_backend import (
    AgentDefinitionBackend,
    MongoAgentBackend,
)
from app.infrastructure.persistence.workflow_backend import (
    LocalFilesWorkflowBackend,
    MongoWorkflowBackend,
    WorkflowDefinitionBackend,
)
from app.infrastructure.tools.mcp_client import McpToolsProvider
from app.infrastructure.triggers.cron_scheduler import CronScheduler

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
    # Factory for per-step LLM overrides; None in legacy test setups.
    llm_factory: Callable[[str | None, str | None], BaseChatModel] | None = None
    # Runners keyed by run_id — alive for the duration of the run so that
    # approval-resume uses the exact definition snapshot from run start.
    live_runners: dict[str, YamlGraphRunner] = field(default_factory=dict)
    cron_scheduler: CronScheduler = field(default_factory=CronScheduler)
    pvc_lease_repository: MongoPvcLeaseRepository | None = None

    async def startup(self) -> None:
        await self.mcp_tools_provider.start()
        self.cron_scheduler.start()
        if self.workflow_backend is not None:
            await self._load_registry()
        asyncio.create_task(self._recover_incomplete_runs())
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
                if self.agent_backend is not None:
                    runner._agent_backend = self.agent_backend
                runner._callback_base_url = self.settings.agent_callback_url or self.settings.base_url
                if self.pvc_lease_repository is not None:
                    runner._pvc_lease_repository = self.pvc_lease_repository
        logger.info("Loaded %d workflow definition(s) from backend", len(definitions))
        self._setup_all_cron_triggers()

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
                )

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
                    if self.agent_backend is not None:
                        runner._agent_backend = self.agent_backend
                    if self.pvc_lease_repository is not None:
                        runner._pvc_lease_repository = self.pvc_lease_repository
                    definition_snapshot: dict | None = defn.to_raw_dict()
                else:
                    runner = self.yaml_graph_registry.get(workflow_id)
                    if runner is None:
                        logger.warning("Cron job: workflow '%s' not found in registry", workflow_id)
                        return
                    definition_snapshot = None

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

                if run.status in ("completed", "failed", "cancelled"):
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
        # Always clear stale cron jobs for this workflow first
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
            if self.agent_backend is not None:
                runner._agent_backend = self.agent_backend
            runner._callback_base_url = self.settings.agent_callback_url or self.settings.base_url
            if self.pvc_lease_repository is not None:
                runner._pvc_lease_repository = self.pvc_lease_repository
            self.yaml_graph_registry._runners[workflow_id] = runner
            self._register_cron_steps(runner)
            logger.info("Registry runner refreshed for workflow '%s'", workflow_id)
        else:
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
                await K8sRuntime(namespace=self.settings.agent_namespace).terminate_by_run_id(run.id)
            except Exception:
                logger.debug("run %s: k8s release cleanup on recovery failed", run.id, exc_info=True)
            try:
                from app.runtime.docker import DockerRuntime
                await DockerRuntime(
                    registry_username=self.settings.docker_registry_username,
                    registry_password=self.settings.docker_registry_password,
                ).terminate_by_run_id(run.id)
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
            if run.status in ("completed", "failed", "cancelled"):
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
                if self.agent_backend is not None:
                    runner._agent_backend = self.agent_backend
                runner._callback_base_url = self.settings.agent_callback_url or self.settings.base_url
                if self.pvc_lease_repository is not None:
                    runner._pvc_lease_repository = self.pvc_lease_repository
                return runner
            except Exception:
                logger.exception("run %s: failed to build runner from definition snapshot", run.id)
        # Fall back to the live registry (e.g. legacy runs without a snapshot)
        return self.yaml_graph_registry.get(run.graph_id)

    async def shutdown(self) -> None:
        self.cron_scheduler.stop()
        await self.mcp_tools_provider.stop()
        await self.mongo_provider.close()
        if self.checkpointer is not None:
            self.checkpointer.close()
        if isinstance(self.workflow_backend, MongoWorkflowBackend):
            await self.workflow_backend.close()
        if isinstance(self.agent_backend, MongoAgentBackend):
            await self.agent_backend.close()


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


def build_container(settings: Settings) -> ApplicationContainer:
    llm = build_llm(settings)
    llm_factory = _make_llm_factory(settings)
    mcp_tools_provider = McpToolsProvider(settings)
    openhands = OpenHandsAdapter(settings)
    mongo_provider = MongoClientProvider(settings)
    run_repository = mongo_provider.get_repository()
    pvc_lease_repository = mongo_provider.get_pvc_lease_repository()
    workflow_backend = _build_workflow_backend(settings)
    # Agent definitions are always stored in MongoDB (no local-files backend).
    agent_backend = MongoAgentBackend(settings.mongodb_uri, settings.mongodb_database)
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
        checkpointer=checkpointer,
        pvc_lease_repository=pvc_lease_repository,
    )
