from __future__ import annotations

import asyncio
import datetime
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel

from app.core.config import Settings
from app.domain.models.graph_run import GraphRun
from app.infrastructure.config.graph_loader import (
    YamlGraphRegistry,
    build_registry_from_definitions,
    build_runner_from_definition,
)
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner, stream_graph_to_pause
from app.infrastructure.persistence.mongo import MongoClientProvider, MongoGraphRunRepository
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
    # Workflow definition backend — None only in legacy test setups that
    # inject the registry directly (backward compat).
    workflow_backend: WorkflowDefinitionBackend | None = None
    # Runners keyed by run_id — alive for the duration of the run so that
    # approval-resume uses the exact definition snapshot from run start.
    live_runners: dict[str, YamlGraphRunner] = field(default_factory=dict)
    cron_scheduler: CronScheduler = field(default_factory=CronScheduler)

    async def startup(self) -> None:
        await self.mcp_tools_provider.start()
        self.cron_scheduler.start()
        if self.workflow_backend is not None:
            await self._load_registry()
        asyncio.create_task(self._recover_incomplete_runs())

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
            mcp_tools_provider=self.mcp_tools_provider,
            openhands=self.openhands,
            run_repository=self.run_repository,
        )
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
                        mcp_tools_provider=self.mcp_tools_provider,
                        registry=self.yaml_graph_registry,
                        run_repository=self.run_repository,
                        openhands=self.openhands,
                    )
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
                mcp_tools_provider=self.mcp_tools_provider,
                registry=self.yaml_graph_registry,
                run_repository=self.run_repository,
                openhands=self.openhands,
            )
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

        config = {"configurable": {"thread_id": run.id}}

        if last_done is not None:
            try:
                runner.graph.update_state(config, accumulated, as_node=last_done)
            except Exception:
                logger.exception("run %s: update_state failed during recovery", run.id)
                run.status = "failed"
                run.state = {"error": "State recovery failed after server restart"}
                run.touch()
                await self.run_repository.update(run)
                return

        # Input for the next astream call: None resumes from checkpoint; dict starts fresh
        resume_input: Any = None if last_done else accumulated

        if run.status == "waiting_approval":
            # Re-execute the approval node to re-arm the interrupt in the new MemorySaver
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
                    mcp_tools_provider=self.mcp_tools_provider,
                    openhands=self.openhands,
                )
                runner._registry = self.yaml_graph_registry
                runner._run_repository = self.run_repository
                return runner
            except Exception:
                logger.exception("run %s: failed to build runner from definition snapshot", run.id)
        # Fall back to the live registry (e.g. legacy runs without a snapshot)
        return self.yaml_graph_registry.get(run.graph_id)

    async def shutdown(self) -> None:
        self.cron_scheduler.stop()
        await self.mcp_tools_provider.stop()
        await self.mongo_provider.close()
        if isinstance(self.workflow_backend, MongoWorkflowBackend):
            await self.workflow_backend.close()


_ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-6"
_OPENAI_DEFAULT_MODEL = "gpt-4o"
_GOOGLE_DEFAULT_MODEL = "gemini-2.0-flash"


def build_llm(settings: Settings) -> BaseChatModel:
    provider = (settings.llm_provider or "").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model or _ANTHROPIC_DEFAULT_MODEL,
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
            max_tokens=16000,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model or _OPENAI_DEFAULT_MODEL,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
            max_tokens=16000,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.llm_model or _GOOGLE_DEFAULT_MODEL,
            google_api_key=settings.google_api_key,  # type: ignore[arg-type]
            max_output_tokens=16000,
        )

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    return FakeMessagesListChatModel(
        responses=[AIMessage(content="LLM not configured. Set LLM_PROVIDER and the matching API key.")]
    )


def _build_workflow_backend(settings: Settings) -> WorkflowDefinitionBackend:
    if settings.workflow_backend_type == "mongodb":
        return MongoWorkflowBackend(settings.mongodb_uri, settings.mongodb_database)
    # Local-files backend: treat every loaded definition as read-only because
    # in production the directory is a k8s ConfigMap volume (readOnly: true).
    return LocalFilesWorkflowBackend(settings.graph_definitions_path, readonly=True)


def build_container(settings: Settings) -> ApplicationContainer:
    llm = build_llm(settings)
    mcp_tools_provider = McpToolsProvider(settings)
    openhands = OpenHandsAdapter(settings)
    mongo_provider = MongoClientProvider(settings)
    run_repository = mongo_provider.get_repository()
    workflow_backend = _build_workflow_backend(settings)
    # Registry starts empty; populated asynchronously in startup().
    return ApplicationContainer(
        settings=settings,
        llm=llm,
        mcp_tools_provider=mcp_tools_provider,
        yaml_graph_registry=YamlGraphRegistry({}),
        mongo_provider=mongo_provider,
        run_repository=run_repository,
        openhands=openhands,
        workflow_backend=workflow_backend,
    )
