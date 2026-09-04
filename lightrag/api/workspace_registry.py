"""Lazy, request-scoped LightRAG resources for multi-workspace API serving."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Callable


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkspaceResources:
    rag: Any
    document_manager: Any


_current_resources: ContextVar[WorkspaceResources | None] = ContextVar(
    "lightrag_current_workspace_resources",
    default=None,
)


class WorkspaceResourceProxy:
    """Forward attribute access to the resources bound to the current request."""

    def __init__(self, default: Any, attribute: str):
        self._default = default
        self._attribute = attribute

    def _target(self) -> Any:
        resources = _current_resources.get()
        if resources is None:
            return self._default
        return getattr(resources, self._attribute)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)


class WorkspaceRegistry:
    """Create one initialized LightRAG resource set per requested workspace."""

    def __init__(
        self,
        default_workspace: str,
        default_resources: WorkspaceResources,
        factory: Callable[[str], WorkspaceResources],
    ) -> None:
        self.default_workspace = default_workspace
        self._factory = factory
        self._resources: dict[str, WorkspaceResources] = {
            default_workspace: default_resources
        }
        self._creation_tasks: dict[str, asyncio.Task[WorkspaceResources]] = {}
        self._lock = asyncio.Lock()
        self._started = False
        self._closing = False

    async def start(self) -> None:
        if self._started:
            return
        await self._initialize(self._resources[self.default_workspace])
        self._started = True

    async def get(self, workspace: str) -> WorkspaceResources:
        if self._closing:
            raise RuntimeError("LightRAG workspace registry is shutting down")

        existing = self._resources.get(workspace)
        if existing is not None:
            return existing

        if not self._started:
            raise RuntimeError("LightRAG workspace registry is not initialized")

        async with self._lock:
            existing = self._resources.get(workspace)
            if existing is not None:
                return existing
            task = self._creation_tasks.get(workspace)
            if task is None:
                task = asyncio.create_task(self._create(workspace))
                self._creation_tasks[workspace] = task

        try:
            resources = await asyncio.shield(task)
        except BaseException:
            async with self._lock:
                if self._creation_tasks.get(workspace) is task and task.done():
                    self._creation_tasks.pop(workspace, None)
            raise

        async with self._lock:
            self._resources[workspace] = resources
            if self._creation_tasks.get(workspace) is task:
                self._creation_tasks.pop(workspace, None)
        return resources

    def bind(self, resources: WorkspaceResources) -> Token:
        return _current_resources.set(resources)

    def reset(self, token: Token) -> None:
        _current_resources.reset(token)

    async def close(self) -> None:
        self._closing = True
        tasks = list(self._creation_tasks.values())
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    logger.error(
                        "Workspace initialization failed during shutdown: %s",
                        result,
                    )

        resources = list(self._resources.values())
        for item in reversed(resources):
            try:
                await item.rag.finalize_storages()
            except Exception as exc:
                logger.error(
                    "Failed to finalize LightRAG workspace '%s': %s",
                    item.rag.workspace,
                    exc,
                )
        self._started = False

    async def _create(self, workspace: str) -> WorkspaceResources:
        resources = self._factory(workspace)
        try:
            await self._initialize(resources)
        except BaseException:
            try:
                await resources.rag.finalize_storages()
            except Exception as exc:
                logger.error(
                    "Failed to finalize partially initialized workspace '%s': %s",
                    workspace,
                    exc,
                )
            raise
        async with self._lock:
            self._resources[workspace] = resources
        return resources

    @staticmethod
    async def _initialize(resources: WorkspaceResources) -> None:
        await resources.rag.initialize_storages()
        await resources.rag.check_and_migrate_data()
