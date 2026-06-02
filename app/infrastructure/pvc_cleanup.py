"""TTL-based PVC cleanup sweeper."""
from __future__ import annotations
import logging
from datetime import datetime, timezone

from app.runtime.pvc_manager import PvcManager

logger = logging.getLogger(__name__)


async def cleanup_expired_pvcs(lease_repo, namespace: str) -> None:
    """Delete all PVCs whose TTL has expired."""
    now = datetime.now(timezone.utc)
    try:
        expired = await lease_repo.get_expired(now)
    except Exception as exc:
        logger.warning("cleanup_expired_pvcs: failed to query leases: %s", exc)
        return
    if not expired:
        return
    mgr = PvcManager(namespace)
    for lease in expired:
        pvc_name = lease.get("pvc_name", "")
        try:
            await mgr.delete_pvc(pvc_name)
            await lease_repo.delete(pvc_name)
            logger.info("cleanup_expired_pvcs: cleaned up PVC %s (run %s)", pvc_name, lease.get("run_id"))
        except Exception as exc:
            logger.warning("cleanup_expired_pvcs: error cleaning %s: %s", pvc_name, exc)
