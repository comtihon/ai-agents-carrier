from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from app.core.config import Settings
from app.infrastructure.config.graph_loader import (
    YamlGraphRegistry,
    build_registry_from_definitions,
)
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.orchestration.yaml_graph import YamlGraphRunner
from app.infrastructure.persistence.mongo import MongoClientProvider, MongoGraphRunRepository
from app.infrastructure.persistence.workflow_backend import (
    LocalFilesWorkflowBackend,
    MongoWorkflowBackend,
    WorkflowDefinitionBackend,
)
from app.infrastructure.tools.mcp_client import McpToolsProvider

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

    async def startup(self) -> None:
        await self.mcp_tools_provider.start()
        if self.workflow_backend is not None:
            await self._load_registry()

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

    async def refresh_runner(self, workflow_id: str) -> None:
        """Rebuild the registry runner for *workflow_id* after a definition change.

        Existing live runners (in-flight runs) are NOT replaced — they keep the
        definition snapshot they started with.
        """
        if self.workflow_backend is None:
            return
        defn = await self.workflow_backend.get(workflow_id)
        if defn is not None:
            from app.infrastructure.config.graph_loader import build_runner_from_definition
            runner = build_runner_from_definition(
                defn,
                llm=self.llm,
                mcp_tools_provider=self.mcp_tools_provider,
                registry=self.yaml_graph_registry,
                run_repository=self.run_repository,
                openhands=self.openhands,
            )
            self.yaml_graph_registry._runners[workflow_id] = runner
            logger.info("Registry runner refreshed for workflow '%s'", workflow_id)
        else:
            self.yaml_graph_registry._runners.pop(workflow_id, None)
            logger.info("Registry runner removed for workflow '%s'", workflow_id)

    async def shutdown(self) -> None:
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
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model or _OPENAI_DEFAULT_MODEL,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.llm_model or _GOOGLE_DEFAULT_MODEL,
            google_api_key=settings.google_api_key,  # type: ignore[arg-type]
        )

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    return FakeMessagesListChatModel(
        responses=[AIMessage(content="LLM not configured. Set LLM_PROVIDER and the matching API key.")]
    )


def _build_workflow_backend(settings: Settings) -> WorkflowDefinitionBackend:
    if settings.workflow_backend_type == "mongodb":
        return MongoWorkflowBackend(settings.mongodb_uri, settings.mongodb_database)
    return LocalFilesWorkflowBackend(settings.graph_definitions_path)


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
