"""Command-line interface for localrag.

Examples:
    python -m app.cli ingest ../sample_docs/*.md
    python -m app.cli ask "How efficient are solar panels?"
    python -m app.cli stats
    python -m app.cli serve
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import RagEngine


def _cmd_ingest(engine: RagEngine, args: argparse.Namespace) -> int:
    total = 0
    for pattern in args.paths:
        matched = [Path(pattern)] if Path(pattern).exists() else list(
            Path().glob(pattern)
        )
        if not matched:
            print(f"! no files matched: {pattern}", file=sys.stderr)
            continue
        for path in matched:
            try:
                result = engine.ingest_file(path)
                print(f"+ ingested {result.source}: {result.n_chunks} chunks")
                total += result.n_chunks
            except Exception as exc:  # noqa: BLE001
                print(f"! failed {path}: {exc}", file=sys.stderr)
    print(f"done. {total} chunks now indexed for these files.")
    return 0


def _cmd_ask(engine: RagEngine, args: argparse.Namespace) -> int:
    ans = engine.answer(args.question, top_k=args.top_k, source=args.source)
    print(f"\n[{ans.mode}{'/' + ans.model if ans.model else ''}] {ans.answer}\n")
    if ans.citations:
        print("Citations:")
        for c in ans.citations:
            print(f"  [{c.marker}] {c.source} (chunk {c.chunk_index}, "
                  f"score {c.score}): {c.snippet}")
    return 0


def _cmd_stats(engine: RagEngine, _args: argparse.Namespace) -> int:
    s = engine.stats()
    print(f"embedder : {s['embedder']} (dim={s['embed_dim']})")
    print(f"chunks   : {s['n_chunks']}")
    print("sources  :")
    for src in s["sources"]:
        print(f"  - {src['source']} ({src['n_chunks']} chunks)")
    return 0


def _cmd_serve(_engine: RagEngine, args: argparse.Namespace) -> int:  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="localrag", description="Offline RAG Q&A.")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest", help="ingest files (md/txt/pdf)")
    pi.add_argument("paths", nargs="+", help="file paths or globs")
    pi.set_defaults(func=_cmd_ingest)

    pa = sub.add_parser("ask", help="ask a question")
    pa.add_argument("question")
    pa.add_argument("--top-k", type=int, default=None)
    pa.add_argument("--source", default=None, help="restrict to one source")
    pa.set_defaults(func=_cmd_ask)

    ps = sub.add_parser("stats", help="show index stats")
    ps.set_defaults(func=_cmd_stats)

    pv = sub.add_parser("serve", help="run the FastAPI server")
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8000)
    pv.set_defaults(func=_cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    engine = RagEngine()
    return args.func(engine, args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
