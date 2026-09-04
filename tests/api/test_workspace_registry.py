import asyncio

import pytest

from lightrag.api.workspace_registry import (
    WorkspaceRegistry,
    WorkspaceResourceProxy,
    WorkspaceResources,
)


class FakeRAG:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self.initialize_calls = 0
        self.migrate_calls = 0
        self.finalize_calls = 0

    async def initialize_storages(self):
        self.initialize_calls += 1

    async def check_and_migrate_data(self):
        self.migrate_calls += 1

    async def finalize_storages(self):
        self.finalize_calls += 1


def make_resources(workspace: str) -> WorkspaceResources:
    return WorkspaceResources(
        rag=FakeRAG(workspace),
        document_manager=type("Manager", (), {"workspace": workspace})(),
    )


@pytest.mark.asyncio
async def test_default_workspace_is_available_before_lifespan_start():
    default = make_resources("default")
    registry = WorkspaceRegistry("default", default, make_resources)

    assert await registry.get("default") is default
    with pytest.raises(RuntimeError, match="not initialized"):
        await registry.get("tenant_chatbot")


@pytest.mark.asyncio
async def test_registry_lazily_initializes_each_workspace_once():
    default = make_resources("default")
    created: list[str] = []

    def factory(workspace: str) -> WorkspaceResources:
        created.append(workspace)
        return make_resources(workspace)

    registry = WorkspaceRegistry("default", default, factory)
    await registry.start()

    first, second = await asyncio.gather(
        registry.get("tenant_chatbot"),
        registry.get("tenant_chatbot"),
    )

    assert first is second
    assert created == ["tenant_chatbot"]
    assert first.rag.initialize_calls == 1
    assert first.rag.migrate_calls == 1

    await registry.close()
    assert default.rag.finalize_calls == 1
    assert first.rag.finalize_calls == 1


@pytest.mark.asyncio
async def test_proxy_uses_request_bound_resources_and_resets_to_default():
    default = make_resources("default")
    selected = make_resources("tenant_chatbot")
    registry = WorkspaceRegistry("default", default, make_resources)
    proxy = WorkspaceResourceProxy(default.rag, "rag")
    await registry.start()

    assert proxy.workspace == "default"
    token = registry.bind(selected)
    try:
        assert proxy.workspace == "tenant_chatbot"
    finally:
        registry.reset(token)
    assert proxy.workspace == "default"

    await registry.close()


@pytest.mark.asyncio
async def test_proxy_keeps_concurrent_workspace_bindings_isolated():
    default = make_resources("default")
    first = make_resources("tenant_one")
    second = make_resources("tenant_two")
    registry = WorkspaceRegistry("default", default, make_resources)
    proxy = WorkspaceResourceProxy(default.rag, "rag")
    await registry.start()

    async def observe(resources: WorkspaceResources) -> str:
        token = registry.bind(resources)
        try:
            await asyncio.sleep(0)
            return proxy.workspace
        finally:
            registry.reset(token)

    assert await asyncio.gather(observe(first), observe(second)) == [
        "tenant_one",
        "tenant_two",
    ]

    await registry.close()
