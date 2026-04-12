from __future__ import annotations

from fastapi import Request

from app.application.services.orchestration_service import OrchestrationService
from app.core.container import ApplicationContainer


def get_container(request: Request) -> ApplicationContainer:
    return request.app.state.container


def get_orchestration_service(request: Request) -> OrchestrationService:
    return get_container(request).orchestration_service
