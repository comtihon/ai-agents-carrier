import logging

logger = logging.getLogger(__name__)


async def cleanup_run_agents(run_id: str, settings) -> None:
    """Terminate all k8s helm releases and docker containers for a run.

    Idempotent and never raises — safe to call on already-cleaned runs.
    """
    try:
        from app.runtime.k8s import K8sRuntime
        await K8sRuntime(namespace=settings.agent_namespace).terminate_by_run_id(run_id)
        logger.info("run %s: k8s agent cleanup done", run_id)
    except Exception:
        logger.debug("run %s: k8s agent cleanup failed", run_id, exc_info=True)
    try:
        from app.runtime.docker import DockerRuntime
        await DockerRuntime(
            registry_username=settings.docker_registry_username,
            registry_password=settings.docker_registry_password,
        ).terminate_by_run_id(run_id)
        logger.info("run %s: docker agent cleanup done", run_id)
    except Exception:
        logger.debug("run %s: docker agent cleanup failed", run_id, exc_info=True)
