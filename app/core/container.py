from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.application.services.orchestration_service import OrchestrationService
from app.application.services.planning_service import PlanningService
from app.core.config import Settings
from app.domain.interfaces.workflow_registry import WorkflowDefinitionRegistry
from app.infrastructure.config.workflow_loader import WorkflowDefinitionLoader
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.orchestration.graph import WorkflowGraphRunner
from app.infrastructure.persistence.mongo import MongoClientProvider, MongoWorkflowRunRepository
from app.infrastructure.actions.http_executor import HttpStepExecutor
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
    graph_runner: WorkflowGraphRunner
    orchestration_service: OrchestrationService

    async def startup(self) -> None:
        await self.mcp_tools_provider.start()

    async def shutdown(self) -> None:
        await self.mcp_tools_provider.stop()
        await self.mongo_provider.close()


def build_container(settings: Settings) -> ApplicationContainer:
    workflow_registry = WorkflowDefinitionLoader(settings.workflow_definitions_path).load()
    mongo_provider = MongoClientProvider(settings)
    workflow_run_repository = mongo_provider.get_repository()
    planning_service = PlanningService()
    tools = build_default_tools()
    mcp_tools_provider = McpToolsProvider(settings)
    action_registry = ActionRegistry()
    http_executor = HttpStepExecutor(timeout=settings.http_action_timeout_seconds)
    openhands_adapter = OpenHandsAdapter(settings)
    graph_runner = WorkflowGraphRunner(
        planning_service,
        openhands_adapter,
        workflow_run_repository,
        mcp_tools_provider,
        http_executor,
        action_registry,
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
        graph_runner=graph_runner,
        orchestration_service=orchestration_service,
    )
