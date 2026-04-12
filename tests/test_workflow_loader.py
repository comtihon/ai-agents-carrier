from pathlib import Path

from app.infrastructure.config.workflow_loader import WorkflowDefinitionLoader


def test_workflow_loader_loads_repository_definitions() -> None:
    registry = WorkflowDefinitionLoader(str(Path("workflows"))).load()
    assert len(registry.list_definitions()) >= 2
    assert registry.get_definition("feature_flow").name == "Feature Delivery Flow"
