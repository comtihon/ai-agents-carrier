from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.application.services.llm_agent_service import LlmAgentService
from app.application.services.orchestration_service import OrchestrationService
from app.application.services.planning_service import PlanningService
from app.core.config import Settings
from app.domain.interfaces.workflow_registry import WorkflowDefinitionRegistry
from app.infrastructure.config.workflow_loader import WorkflowDefinitionLoader
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.orchestration.graph import WorkflowGraphRunner
from app.infrastructure.persistence.mongo import MongoClientProvider, MongoWorkflowRunRepository
from app.infrastructure.actions.http_executor import HttpStepExecutor
from app.infrastructure.actions.loader import load_actions_from_directory
from app.infrastructure.actions.registry import ActionRegistry
from app.infrastructure.tools.langchain_tools import build_default_tools
from app.infrastructure.tools.mcp_client import McpToolsProvider


@dataclass
class ApplicationContainer:
    settings: Settings
    mongo_provider: MongoClientProvider
    workflow_registry: WorkflowDefinitionRegistry
    workflow_run_repository: MongoWorkflowRunRepository
    planning_service: PlanningService
    tools: list[Any]
    mcp_tools_provider: McpToolsProvider
    action_registry: ActionRegistry
    llm_agent_service: LlmAgentService
    graph_runner: WorkflowGraphRunner
    orchestration_service: OrchestrationService

    async def startup(self) -> None:
        await self.mcp_tools_provider.start()

    async def shutdown(self) -> None:
        await self.mcp_tools_provider.stop()
        await self.mongo_provider.close()


_ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-6"
_OPENAI_DEFAULT_MODEL = "gpt-4o"


def build_llm(settings: Settings) -> BaseChatModel:
    """Instantiate the configured chat model.

    Reads LLM_PROVIDER, LLM_MODEL, ANTHROPIC_API_KEY, and OPENAI_API_KEY from
    settings.  When no provider is configured a lightweight no-op stub is
    returned so the service starts without requiring credentials — any workflow
    that reaches an ``llm`` step will fail with a clear message.
    """
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

    # No provider configured — return a stub that fails loudly on first use
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    return FakeMessagesListChatModel(
        responses=[AIMessage(content="LLM not configured. Set LLM_PROVIDER and the matching API key.")]
    )


def build_container(settings: Settings) -> ApplicationContainer:
    workflow_registry = WorkflowDefinitionLoader(settings.workflow_definitions_path).load()
    mongo_provider = MongoClientProvider(settings)
    workflow_run_repository = mongo_provider.get_repository()
    planning_service = PlanningService()
    tools = build_default_tools()
    mcp_tools_provider = McpToolsProvider(settings)
    action_registry = ActionRegistry()
    load_actions_from_directory(settings.workflow_definitions_path, action_registry)
    http_executor = HttpStepExecutor(timeout=settings.http_action_timeout_seconds)
    openhands_adapter = OpenHandsAdapter(settings)
    llm_agent_service = LlmAgentService(llm=build_llm(settings), tools=tools)
    graph_runner = WorkflowGraphRunner(
        planning_service,
        openhands_adapter,
        workflow_run_repository,
        mcp_tools_provider,
        http_executor,
        action_registry,
        llm_agent_service,
    )
    orchestration_service = OrchestrationService(workflow_registry, workflow_run_repository, graph_runner)
    return ApplicationContainer(
        settings=settings,
        mongo_provider=mongo_provider,
        workflow_registry=workflow_registry,
        workflow_run_repository=workflow_run_repository,
        planning_service=planning_service,
        tools=tools,
        mcp_tools_provider=mcp_tools_provider,
        action_registry=action_registry,
        llm_agent_service=llm_agent_service,
        graph_runner=graph_runner,
        orchestration_service=orchestration_service,
    )
