"""The workspace of a runner agent: pre-signed URLs acp-web-proxy syncs it with.

A pod agent restores its workspace from the agent's ``s3`` addon itself
(pi-cloud-agent, with the pod's bucket access). A runner is a user's machine
with no carrier credentials, so carrier hands its proxy a V4-signed GET and PUT
URL for the same object instead — ``gs://<bucket>/<path>/workspace.tar.gz`` —
in ``_proxy/launch`` ``config.workspace``. The proxy restores the session
directory from the one before the agent starts and uploads it to the other
after every turn, so pod and runner steps of one run share one workspace.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Same object pi-cloud-agent restores from and uploads to.
ARCHIVE_NAME = "workspace.tar.gz"
ARCHIVE_TYPE = "application/gzip"
# V4 signed URLs are valid for at most seven days.
MAX_URL_TTL_S = 7 * 24 * 3600


class WorkspaceUnavailable(Exception):
    """The workspace cannot be synced; the message says why."""


async def runner_workspace(agent_config: dict[str, Any], cwd: str, settings: Any) -> tuple[dict[str, Any], str] | None:
    """``config.workspace`` for ``_proxy/launch`` and the archive's gs:// URI.

    ``None`` when the agent has no ``s3`` addon. Raises WorkspaceUnavailable
    when it has one but carrier cannot sign URLs for it.
    """
    extra = agent_config.get("extra") or {}
    bucket = str(extra.get("s3_bucket") or "").strip()
    path = str(extra.get("s3_path") or "").strip("/")
    if not bucket or not path:
        return None
    obj = f"{path}/{ARCHIVE_NAME}"
    ttl = min(int(getattr(settings, "acp_runner_workspace_url_ttl_s", 86400)), MAX_URL_TTL_S)
    try:
        download_url, upload_url = await asyncio.to_thread(_sign_urls, bucket, obj, ttl)
    except Exception as exc:  # noqa: BLE001 — every failure means "cannot sign"
        logger.warning("cannot sign workspace URLs for gs://%s/%s: %s", bucket, obj, exc)
        raise WorkspaceUnavailable(
            f"carrier cannot sign URLs for gs://{bucket}/{obj} ({exc}); under Workload Identity its "
            "service account needs roles/iam.serviceAccountTokenCreator on itself"
        ) from exc
    spec: dict[str, Any] = {
        "dir": cwd,
        "download_url": download_url,
        "upload_url": upload_url,
        "upload_headers": {"content-type": ARCHIVE_TYPE},
    }
    exclude = extra.get("workspace_exclude")
    if isinstance(exclude, list) and exclude:
        spec["exclude"] = [str(e) for e in exclude if e]
    return spec, f"gs://{bucket}/{obj}"


def _sign_urls(bucket_name: str, object_name: str, ttl_s: int) -> tuple[str, str]:
    """A signed GET and PUT URL for one object (blocking)."""
    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(object_name)
    try:
        return _sign_pair(blob, ttl_s)
    except AttributeError:
        # No private key (Workload Identity): sign with the IAM signBlob API.
        import google.auth.transport.requests

        creds = client._credentials
        creds.refresh(google.auth.transport.requests.Request())
        return _sign_pair(blob, ttl_s, service_account_email=creds.service_account_email, access_token=creds.token)


def _sign_pair(blob: Any, ttl_s: int, **signer: Any) -> tuple[str, str]:
    expiration = timedelta(seconds=ttl_s)
    get = blob.generate_signed_url(version="v4", expiration=expiration, method="GET", **signer)
    put = blob.generate_signed_url(version="v4", expiration=expiration, method="PUT", content_type=ARCHIVE_TYPE, **signer)
    return get, put
