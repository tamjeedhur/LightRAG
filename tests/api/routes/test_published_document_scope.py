import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
document_routes = importlib.import_module("lightrag.api.routers.document_routes")
query_routes = importlib.import_module("lightrag.api.routers.query_routes")
sys.argv = _argv


@pytest.mark.parametrize("ids", [[], ["doc-a"], ["doc-a", "doc-b"]])
def test_query_preserves_publication_scope(ids):
    request = query_routes.QueryRequest(query="policy", mode="naive", document_ids=ids)
    assert request.to_query_params(False).document_ids == ids


def test_query_rejects_graph_scope_instead_of_silently_ignoring_it():
    with pytest.raises(ValueError, match="requires naive"):
        query_routes.QueryRequest(query="policy", mode="mix", document_ids=["doc-a"])


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [[], ["doc-a"], ["doc-a", "doc-b"]])
async def test_data_query_preserves_scope_through_public_entry_point(monkeypatch, ids):
    from lightrag import LightRAG
    from lightrag.base import QueryParam
    import lightrag.lightrag as rag_module

    rag = object.__new__(LightRAG)
    rag._build_global_config = Mock(return_value={})
    rag.chunks_vdb = SimpleNamespace()
    rag.llm_response_cache = None
    rag.text_chunks = None
    rag._query_done = AsyncMock()
    query = AsyncMock(return_value=None)
    monkeypatch.setattr(rag_module, "naive_query", query)
    param = QueryParam(mode="naive", document_ids=ids)
    result = await rag.aquery_data("policy", param)
    assert result["status"] == "failure"
    assert query.await_args.args[2].document_ids == ids
    assert param.only_need_context is False


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [[], ["doc-a"]])
async def test_naive_vector_context_applies_scope_and_keeps_document_identity(ids):
    from lightrag.base import QueryParam
    from lightrag.operate import _get_vector_context

    storage = SimpleNamespace(
        cosine_better_than_threshold=0.1,
        query=AsyncMock(return_value=[] if not ids else [{"id": "chunk-a", "full_doc_id": "doc-a", "content": "policy", "file_path": "cb-a"}]),
    )
    result = await _get_vector_context("policy", storage, QueryParam(mode="naive", document_ids=ids), None)
    assert storage.query.await_args.kwargs["document_ids"] == ids
    if ids:
        assert result[0]["full_doc_id"] == "doc-a"
    else:
        assert result == []


@pytest.mark.parametrize("record", [None, {"status": "failed", "file_path": "cb-a"}, {"status": "processed", "file_path": "cb-a", "chunks_count": 7}])
def test_exact_document_lookup_uses_scoped_storage(record):
    storage = SimpleNamespace(get_by_id=AsyncMock(return_value=record))
    app = FastAPI()
    app.include_router(document_routes.create_document_routes(SimpleNamespace(doc_status=storage), SimpleNamespace(), api_key="test-key"))
    response = TestClient(app).get("/documents/status/doc-a", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert response.json() == ({"id": "doc-a", **record} if record else None)
    storage.get_by_id.assert_awaited_once_with("doc-a")
