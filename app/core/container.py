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
from app.infrastructure.tools.langchain_tools import build_default_tools


@dataclass
class ApplicationContainer:
    settings: Settings
    mongo_provider: MongoClientProvider
    workflow_registry: WorkflowDefinitionRegistry
    workflow_run_repository: MongoWorkflowRunRepository
    planning_service: PlanningService
    tools: list[Any]
    graph_runner: WorkflowGraphRunner
    orchestration_service: OrchestrationService

    async def shutdown(self) -> None:
        await self.mongo_provider.close()


def build_container(settings: Settings) -> ApplicationContainer:
    workflow_registry = WorkflowDefinitionLoader(settings.workflow_definitions_path).load()
    mongo_provider = MongoClientProvider(settings)
    workflow_run_repository = mongo_provider.get_repository()
    planning_service = PlanningService()
    tools = build_default_tools()
    openhands_adapter = OpenHandsAdapter(settings)
    graph_runner = WorkflowGraphRunner(planning_service, openhands_adapter, workflow_run_repository)
    orchestration_service = OrchestrationService(workflow_registry, workflow_run_repository, graph_runner)
    return ApplicationContainer(
        settings=settings,
        mongo_provider=mongo_provider,
        workflow_registry=workflow_registry,
        workflow_run_repository=workflow_run_repository,
        planning_service=planning_service,
        tools=tools,
        graph_runner=graph_runner,
        orchestration_service=orchestration_service,
    )
