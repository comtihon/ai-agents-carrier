"""Vendored MongoDB checkpoint saver for LangGraph.

Based on langgraph-checkpoint-mongodb (MIT licence) but without the
langchain-mongodb / numpy / sqlalchemy transitive dependency chain.
Compatible with langgraph>=1.1.0 and pymongo>=4.12.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig, run_in_executor
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database as MongoDatabase


def _loads_metadata(serde: SerializerProtocol, metadata: Any) -> Any:
    if isinstance(metadata, dict):
        return {k: _loads_metadata(serde, v) for k, v in metadata.items()}
    return serde.loads_typed(metadata)


def _dumps_metadata(serde: SerializerProtocol, metadata: Any) -> Any:
    if isinstance(metadata, dict):
        return {k: _dumps_metadata(serde, v) for k, v in metadata.items()}
    return serde.dumps_typed(metadata)


def _ensure_indexes(
    collection: Collection,
    compound_index: list[tuple[str, int]],
    ttl: Optional[int] = None,
) -> None:
    def key_list(idx: Any) -> list[tuple[str, int]]:
        return list((k, v) for k, v in idx["key"].items())

    existing = [key_list(i) for i in collection.list_indexes()]
    if compound_index not in existing:
        collection.create_index(compound_index, unique=True)
    if ttl is not None:
        ttl_key = [("created_at", ASCENDING)]
        if not any(key_list(i) == ttl_key and i.get("expireAfterSeconds") == ttl
                   for i in collection.list_indexes()):
            collection.create_index(ttl_key, expireAfterSeconds=ttl)


class MongoDBCheckpointSaver(BaseCheckpointSaver):
    """Synchronous MongoDB checkpoint saver backed by pymongo.

    Async methods (`aget_tuple`, `aput`, `aput_writes`) delegate to their
    sync counterparts via `run_in_executor`, making them safe for use inside
    FastAPI async handlers and LangGraph's `astream` / `ainvoke`.

    Usage::

        from pymongo import MongoClient
        client = MongoClient(settings.mongodb_uri)
        checkpointer = MongoDBCheckpointSaver(client, db_name="mydb")
        graph = sg.compile(checkpointer=checkpointer)
    """

    client: MongoClient
    db: MongoDatabase

    def __init__(
        self,
        client: MongoClient,
        db_name: str = "langgraph",
        checkpoint_collection_name: str = "lg_checkpoints",
        writes_collection_name: str = "lg_checkpoint_writes",
        ttl: Optional[int] = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.db = client[db_name]
        self.checkpoints: Collection = self.db[checkpoint_collection_name]
        self.writes: Collection = self.db[writes_collection_name]
        self.ttl = ttl
        self.serde = serde or JsonPlusSerializer()

        _ensure_indexes(
            self.checkpoints,
            [("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1)],
            self.ttl,
        )
        _ensure_indexes(
            self.writes,
            [("thread_id", 1), ("checkpoint_ns", 1), ("checkpoint_id", -1),
             ("task_id", 1), ("idx", 1)],
            self.ttl,
        )

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        db_name: str = "langgraph",
        **kwargs: Any,
    ) -> Iterator["MongoDBCheckpointSaver"]:
        client: Optional[MongoClient] = None
        try:
            client = MongoClient(conn_string)
            yield cls(client, db_name=db_name, **kwargs)
        finally:
            if client:
                client.close()

    def close(self) -> None:
        self.client.close()

    # ── Sync interface ──────────────────────────────────────────────────────

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id := get_checkpoint_id(config):
            query = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns,
                     "checkpoint_id": checkpoint_id}
        else:
            query = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}

        result = self.checkpoints.find(query, sort=[("checkpoint_id", -1)], limit=1)
        for doc in result:
            cfg_vals = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": doc["checkpoint_id"],
            }
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            pending_writes = [
                (w["task_id"], w["channel"],
                 self.serde.loads_typed((w["type"], w["value"])))
                for w in self.writes.find(cfg_vals)
            ]
            return CheckpointTuple(
                {"configurable": cfg_vals},
                checkpoint,
                _loads_metadata(self.serde, doc["metadata"]),
                (
                    {"configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": doc["parent_checkpoint_id"],
                    }}
                    if doc.get("parent_checkpoint_id") else None
                ),
                pending_writes,
            )
        return None

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        query: dict[str, Any] = {}
        if config is not None:
            cfg = config["configurable"]
            if "thread_id" in cfg:
                query["thread_id"] = cfg["thread_id"]
            if "checkpoint_ns" in cfg:
                query["checkpoint_ns"] = cfg["checkpoint_ns"]
        if filter:
            for key, value in filter.items():
                query[f"metadata.{key}"] = _dumps_metadata(self.serde, value)
        if before is not None:
            query["checkpoint_id"] = {"$lt": before["configurable"]["checkpoint_id"]}

        cursor = self.checkpoints.find(
            query,
            limit=0 if limit is None else limit,
            sort=[("checkpoint_id", -1)],
        )
        for doc in cursor:
            cfg_vals = {
                "thread_id": doc["thread_id"],
                "checkpoint_ns": doc["checkpoint_ns"],
                "checkpoint_id": doc["checkpoint_id"],
            }
            pending_writes = [
                (w["task_id"], w["channel"],
                 self.serde.loads_typed((w["type"], w["value"])))
                for w in self.writes.find(cfg_vals)
            ]
            yield CheckpointTuple(
                config={"configurable": cfg_vals},
                checkpoint=self.serde.loads_typed((doc["type"], doc["checkpoint"])),
                metadata=_loads_metadata(self.serde, doc["metadata"]),
                parent_config=(
                    {"configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc["checkpoint_ns"],
                        "checkpoint_id": doc["parent_checkpoint_id"],
                    }}
                    if doc.get("parent_checkpoint_id") else None
                ),
                pending_writes=pending_writes,
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]

        type_, serialized = self.serde.dumps_typed(checkpoint)
        meta = dict(metadata)
        meta.update(config.get("metadata", {}))
        doc: dict[str, Any] = {
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "type": type_,
            "checkpoint": serialized,
            "metadata": _dumps_metadata(self.serde, meta),
        }
        if self.ttl:
            doc["created_at"] = datetime.now(tz=timezone.utc)

        self.checkpoints.update_one(
            {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns,
             "checkpoint_id": checkpoint_id},
            {"$set": doc},
            upsert=True,
        )
        return {"configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }}

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        set_method = "$set" if all(w[0] in WRITES_IDX_MAP for w in writes) else "$setOnInsert"
        ops = []
        now = datetime.now(tz=timezone.utc)
        for idx, (channel, value) in enumerate(writes):
            type_, serialized = self.serde.dumps_typed(value)
            update_doc: dict[str, Any] = {"channel": channel, "type": type_, "value": serialized}
            if self.ttl:
                update_doc["created_at"] = now
            ops.append(UpdateOne(
                filter={"thread_id": thread_id, "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id, "task_id": task_id,
                        "task_path": task_path,
                        "idx": WRITES_IDX_MAP.get(channel, idx)},
                update={set_method: update_doc},
                upsert=True,
            ))
        if ops:
            self.writes.bulk_write(ops)

    def delete_thread(self, thread_id: str) -> None:
        self.checkpoints.delete_many({"thread_id": thread_id})
        self.writes.delete_many({"thread_id": thread_id})

    # ── Async wrappers ──────────────────────────────────────────────────────

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await run_in_executor(None, self.get_tuple, config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def _run() -> None:
            try:
                for item in self.list(config, filter=filter, before=before, limit=limit):
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        await run_in_executor(None, _run)
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await run_in_executor(None, self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return await run_in_executor(None, self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return await run_in_executor(None, self.delete_thread, thread_id)
