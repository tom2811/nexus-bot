"""Tests for multi-tenant document ingest."""

from pathlib import Path
from unittest.mock import MagicMock

from ingest import ingest_tenant, main


def test_ingest_skips_when_collection_already_has_docs(tmp_path, monkeypatch):
    store = MagicMock()
    store.count.return_value = 6
    monkeypatch.setattr("ingest.VectorStore", lambda **kwargs: store)

    ingest_tenant(tmp_path, "kickskraze")

    store.add_documents.assert_not_called()


def test_main_ingests_only_selected_tenants(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    (data_root / "docs").mkdir(parents=True)
    (data_root / "kickskraze").mkdir()
    monkeypatch.setenv("TENANT_IDS", "kickskraze")

    called = []

    def fake_ingest(tenant_dir: Path, collection_name: str, persist_dir: str = "./chroma_db"):
        called.append(collection_name)

    monkeypatch.setattr("ingest.ingest_tenant", fake_ingest)
    main(data_root=data_root, persist_dir=str(tmp_path / "chroma"))

    assert called == ["kickskraze"]
