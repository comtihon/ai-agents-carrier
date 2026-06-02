"""Kubernetes PVC lifecycle management for agent pods."""
from __future__ import annotations
import asyncio
import logging
import re
from datetime import timedelta

logger = logging.getLogger(__name__)

_VALID_NAME = re.compile(r'^[a-z0-9][a-z0-9\-]{0,251}[a-z0-9]$|^[a-z0-9]$')


def is_valid_pvc_name(name: str) -> bool:
    return bool(_VALID_NAME.match(name))


def parse_ttl(ttl_str: str) -> timedelta:
    """Parse e.g. '30m', '1h', '2h', '7d' → timedelta. Defaults to 1h on error."""
    ttl_str = (ttl_str or "").strip().lower()
    try:
        if ttl_str.endswith('d'):
            return timedelta(days=int(ttl_str[:-1]))
        if ttl_str.endswith('h'):
            return timedelta(hours=int(ttl_str[:-1]))
        if ttl_str.endswith('m'):
            return timedelta(minutes=int(ttl_str[:-1]))
    except (ValueError, IndexError):
        pass
    logger.warning("Invalid TTL %r, defaulting to 1h", ttl_str)
    return timedelta(hours=1)


class PvcManager:
    def __init__(self, namespace: str) -> None:
        self._namespace = namespace
        self._client = None

    def _get_core_v1(self):
        if self._client is not None:
            return self._client
        try:
            from kubernetes import client, config  # type: ignore
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
            self._client = client.CoreV1Api()
            return self._client
        except Exception as exc:
            logger.warning("PvcManager: k8s client unavailable: %s", exc)
            return None

    async def create_pvc(self, name: str, storage: str = "1Gi") -> bool:
        """Create PVC if it doesn't exist. Returns True if created/exists."""
        core = self._get_core_v1()
        if core is None:
            return False
        def _create():
            from kubernetes import client  # type: ignore
            body = client.V1PersistentVolumeClaim(
                metadata=client.V1ObjectMeta(name=name),
                spec=client.V1PersistentVolumeClaimSpec(
                    access_modes=["ReadWriteOnce"],
                    resources=client.V1ResourceRequirements(
                        requests={"storage": storage}
                    ),
                ),
            )
            try:
                core.create_namespaced_persistent_volume_claim(self._namespace, body)
                logger.info("PvcManager: created PVC %s/%s", self._namespace, name)
                return True
            except Exception as exc:
                # 409 = already exists — treat as success
                if hasattr(exc, 'status') and exc.status == 409:
                    return True
                logger.warning("PvcManager: failed to create PVC %s: %s", name, exc)
                return False
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _create)

    async def delete_pvc(self, name: str) -> None:
        """Delete PVC. 404 treated as success (idempotent)."""
        core = self._get_core_v1()
        if core is None:
            return
        def _delete():
            try:
                core.delete_namespaced_persistent_volume_claim(name, self._namespace)
                logger.info("PvcManager: deleted PVC %s/%s", self._namespace, name)
            except Exception as exc:
                if hasattr(exc, 'status') and exc.status == 404:
                    return
                logger.warning("PvcManager: failed to delete PVC %s: %s", name, exc)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _delete)
