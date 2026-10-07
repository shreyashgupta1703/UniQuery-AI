"""CLI tests: ingest / ask / stats over an in-memory engine.

Drives the argparse entrypoints directly with a hermetic engine (hashing
embedder, in-memory DB) so the command surface is covered without touching the
network, a model, or disk.
"""

from __future__ import annotations

import pytest

from app import cli
from app.embeddings import HashingEmbedder
from app.engine import RagEngine
from app.store import VectorStore


@pytest.fixture()
def engine(monkeypatch):
    eng = RagEngine(embedder=HashingEmbedder(dim=256), store=VectorStore(":memory:"))
    eng.ingest_text(
        "Solar panels convert sunlight into electricity. Efficiency is 18 to 22 "
        "percent for silicon panels.",
        "solar.md",
    )
    # Force the extractive path regardless of any Ollama on the machine.
    monkeypatch.setattr(cli, "RagEngine", lambda *a, **k: eng)
    return eng


def test_cli_ask(engine, capsys):
    parser = cli.build_parser()
    args = parser.parse_args(["ask", "How efficient are solar panels?"])
    rc = args.func(engine, args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "[extractive" in out
    assert "Citations:" in out
    assert "solar.md" in out


def test_cli_stats(engine, capsys):
    parser = cli.build_parser()
    args = parser.parse_args(["stats"])
    rc = args.func(engine, args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "embedder" in out
    assert "solar.md" in out


def test_cli_ingest_missing_file(engine, capsys):
    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "does_not_exist_*.md"])
    rc = args.func(engine, args)
    assert rc == 0  # missing globs are reported, not fatal
    err = capsys.readouterr().err
    assert "no files matched" in err


def test_cli_main_dispatches(engine, capsys):
    # main() builds its own engine, but we've patched RagEngine to our fixture.
    rc = cli.main(["stats"])
    assert rc == 0
    assert "chunks" in capsys.readouterr().out
