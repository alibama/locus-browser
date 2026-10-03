import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def ollama():
    import fake_ollama
    srv, url = fake_ollama.start()
    yield url
    srv.shutdown()


@pytest.fixture()
def env(tmp_path, monkeypatch, ollama):
    import make_synthetic
    pq = make_synthetic.make(tmp_path / "syn.parquet")
    monkeypatch.setenv("LOCUS_SRC", str(pq))
    monkeypatch.setenv("LOCUS_SLIM", str(tmp_path / "none.parquet"))
    monkeypatch.setenv("LOCUS_DB", str(tmp_path / "t.db"))
    monkeypatch.setenv("OLLAMA_HOST", ollama)
    return {"pq": pq, "host": ollama, "model": "qwen2.5:7b"}
