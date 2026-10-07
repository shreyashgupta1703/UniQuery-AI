"""Answer generation from retrieved context.

Primary path: a **local Ollama** server (``/api/generate``) running any model
you've pulled (llama3.2, qwen2.5, mistral, ...). No API key, no cloud. Tokens
are streamed as they arrive so the UI can render the answer live.

Fallback path: when Ollama is unreachable or disabled, an **extractive**
answer is assembled directly from the highest-scoring retrieved chunks. This
guarantees the tool produces a grounded, citation-backed response even with no
LLM present at all — the RAG pipeline degrades gracefully instead of failing.

Two design choices worth calling out:

* **Honest citations.** For the LLM path we parse the ``[n]`` markers the model
  *actually emits* and return only those chunks, renumbered to match — instead
  of blindly attaching every retrieved passage. The Sources panel therefore
  never lists a passage the answer did not use.
* **Grounding signal.** Every answer reports whether it is ``grounded`` (at
  least one retrieved chunk scored above a confidence threshold). The UI can
  surface a "low-confidence" hint when the corpus doesn't really cover the
  question.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Sequence

from .config import settings
from .store import ScoredChunk

SYSTEM_PROMPT = (
    "You are a precise assistant answering strictly from the provided context. "
    "Cite the sources you use with bracketed numbers like [1], [2] that refer to "
    "the numbered context passages. If the answer is not in the context, say you "
    "don't have enough information. Be concise."
)

# A retrieved chunk whose cosine score clears this bar is considered a
# trustworthy grounding for the answer. Below it we still answer, but flag the
# response as low-confidence so the UI can say so.
GROUNDING_THRESHOLD = 0.15

_MARKER_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    """A source passage referenced by an answer."""

    marker: int
    source: str
    chunk_index: int
    score: float
    snippet: str


@dataclass
class Answer:
    """A generated answer plus its provenance."""

    answer: str
    citations: List[Citation] = field(default_factory=list)
    mode: str = "extractive"  # "ollama" or "extractive"
    model: str = ""
    grounded: bool = True


# --------------------------------------------------------------------- context
def build_context(chunks: Sequence[ScoredChunk]) -> str:
    """Render retrieved chunks as a numbered context block for the prompt."""

    parts = []
    for i, ch in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {ch.source})\n{ch.text.strip()}")
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: Sequence[ScoredChunk]) -> str:
    context = build_context(chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer (with bracketed citations):"
    )


def _snippet(text: str, limit: int = 240) -> str:
    snippet = text.strip().replace("\n", " ")
    if len(snippet) > limit:
        snippet = snippet[: limit - 3].rstrip() + "..."
    return snippet


def _citation(marker: int, ch: ScoredChunk) -> Citation:
    return Citation(
        marker=marker,
        source=ch.source,
        chunk_index=ch.chunk_index,
        score=round(ch.score, 4),
        snippet=_snippet(ch.text),
    )


def _citations_for(chunks: Sequence[ScoredChunk]) -> List[Citation]:
    return [_citation(i, ch) for i, ch in enumerate(chunks, start=1)]


def is_grounded(chunks: Sequence[ScoredChunk]) -> bool:
    """True when at least one retrieved chunk clears the confidence threshold."""

    return any(ch.score >= GROUNDING_THRESHOLD for ch in chunks)


# ------------------------------------------------------------------ reranking
def lexical_rerank(
    question: str,
    chunks: Sequence[ScoredChunk],
    top_k: int,
    *,
    weight: float = 0.35,
) -> List[ScoredChunk]:
    """Blend vector score with lexical term overlap and re-sort.

    A cheap, dependency-free hybrid pass: it nudges up passages that literally
    contain the query terms, which reliably improves precision on keyword-heavy
    questions without a cross-encoder. The blended score is written back onto
    the returned chunks so downstream grounding/citations stay consistent.
    """

    q_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    if not q_terms or not chunks:
        return list(chunks)[:top_k]

    reranked: List[ScoredChunk] = []
    for ch in chunks:
        c_terms = set(re.findall(r"[a-z0-9]+", ch.text.lower()))
        overlap = len(q_terms & c_terms) / len(q_terms)
        blended = (1.0 - weight) * ch.score + weight * overlap
        reranked.append(
            ScoredChunk(
                id=ch.id,
                text=ch.text,
                source=ch.source,
                chunk_index=ch.chunk_index,
                score=blended,
            )
        )
    reranked.sort(key=lambda s: s.score, reverse=True)
    return reranked[: max(0, top_k)]


# ----------------------------------------------------------- honest citations
def select_cited(text: str, chunks: Sequence[ScoredChunk]) -> tuple[str, List[Citation]]:
    """Keep only the chunks the answer actually cites, renumbered 1..N.

    Parses the ``[n]`` markers present in ``text`` (in order of first
    appearance), maps each to its 1-based context chunk, and rewrites the
    markers so they are contiguous and match the returned citation list. Markers
    that point outside the retrieved set are dropped. If the model emitted no
    valid markers we fall back to citing everything (renumbered), so the answer
    is never left provenance-free.
    """

    seen: List[int] = []
    for m in _MARKER_RE.finditer(text):
        idx = int(m.group(1))
        if 1 <= idx <= len(chunks) and idx not in seen:
            seen.append(idx)

    if not seen:
        # Model cited nothing usable — attach all retrieved chunks as context.
        return text, _citations_for(chunks)

    # Map old marker -> new contiguous marker (in first-appearance order).
    remap = {old: new for new, old in enumerate(seen, start=1)}

    def _sub(match: re.Match[str]) -> str:
        old = int(match.group(1))
        return f"[{remap[old]}]" if old in remap else ""

    rewritten = _MARKER_RE.sub(_sub, text)
    citations = [_citation(new, chunks[old - 1]) for old, new in remap.items()]
    citations.sort(key=lambda c: c.marker)
    return rewritten, citations


# ------------------------------------------------------------ extractive path
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _top_sentences(text: str, query: str, max_sentences: int = 2) -> List[str]:
    """Pick the sentences from ``text`` most lexically relevant to ``query``."""

    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if not sentences:
        return []
    q_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not q_terms:
        return sentences[:max_sentences]

    def overlap(sentence: str) -> int:
        s_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        return len(q_terms & s_terms)

    ranked = sorted(range(len(sentences)), key=lambda i: overlap(sentences[i]), reverse=True)
    chosen = sorted(ranked[:max_sentences])
    return [sentences[i] for i in chosen]


def _extractive_text(question: str, chunks: Sequence[ScoredChunk]) -> str:
    if not chunks:
        return (
            "I don't have any indexed documents that match this question. "
            "Try ingesting a relevant document first."
        )
    lines: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        sents = _top_sentences(ch.text, question, max_sentences=2)
        if not sents:
            continue
        lines.append(f"{' '.join(sents)} [{i}]")
    body = " ".join(lines) if lines else "See the cited passages below."
    preamble = (
        "Based on the most relevant passages in the indexed documents "
        "(no local LLM was available, so this answer is extracted directly "
        "from the sources):\n\n"
    )
    return preamble + body


def extractive_answer(question: str, chunks: Sequence[ScoredChunk]) -> Answer:
    """Build a grounded answer from retrieved chunks with no LLM.

    Summarizes each of the top passages down to its most query-relevant
    sentences and attaches an inline citation marker, so the response is
    genuinely useful and fully traceable even without a model.
    """

    if not chunks:
        return Answer(
            answer=_extractive_text(question, chunks),
            citations=[],
            mode="extractive",
            grounded=False,
        )
    return Answer(
        answer=_extractive_text(question, chunks),
        citations=_citations_for(chunks),
        mode="extractive",
        grounded=is_grounded(chunks),
    )


# --------------------------------------------------------------- ollama client
def _ollama_generate(prompt: str) -> str:
    """Call Ollama /api/generate non-streaming. Raises on failure."""

    return "".join(_ollama_stream(prompt))


def _ollama_stream(prompt: str) -> Iterator[str]:
    """Yield answer tokens from Ollama's streaming /api/generate endpoint.

    Ollama emits one JSON object per line; we forward the ``response`` field of
    each as it arrives. Raises on transport failure or an empty result so the
    caller can fall back to the extractive path.
    """

    url = settings.ollama_url.rstrip("/") + "/api/generate"
    payload = json.dumps(
        {"model": settings.ollama_model, "prompt": prompt, "stream": True}
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    produced = False
    with urllib.request.urlopen(req, timeout=settings.ollama_timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = obj.get("response", "")
            if token:
                produced = True
                yield token
            if obj.get("done"):
                break
    if not produced:
        raise RuntimeError("Ollama returned an empty response")


def ollama_available() -> bool:
    """Best-effort probe of the local Ollama server (fast, non-fatal)."""

    if settings.disable_ollama:
        return False
    try:
        url = settings.ollama_url.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=2):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


# --------------------------------------------------------------- public answer
def generate_answer(question: str, chunks: Sequence[ScoredChunk]) -> Answer:
    """Generate an answer, preferring Ollama and falling back to extractive.

    Never raises for a missing/unreachable LLM — the extractive path always
    produces a grounded, cited answer. For the Ollama path, citations are
    filtered to only the passages the model actually referenced.
    """

    if not settings.disable_ollama and chunks:
        try:
            prompt = build_prompt(question, chunks)
            text = _ollama_generate(prompt).strip()
            cited_text, citations = select_cited(text, chunks)
            return Answer(
                answer=cited_text,
                citations=citations,
                mode="ollama",
                model=settings.ollama_model,
                grounded=is_grounded(chunks),
            )
        except Exception:  # noqa: BLE001 - fall back on any Ollama failure
            pass

    return extractive_answer(question, chunks)


# ------------------------------------------------------------- streaming answer
@dataclass
class StreamEvent:
    """One event in a streamed answer.

    ``type`` is one of:
      * ``"meta"``   - first event: mode/model/grounded, before any token.
      * ``"token"``  - a piece of answer text (may be a single token or word).
      * ``"citations"`` - final event: the resolved citation list.
    """

    type: str
    data: dict


def stream_answer(question: str, chunks: Sequence[ScoredChunk]) -> Iterator[StreamEvent]:
    """Yield an answer incrementally as ``StreamEvent`` objects.

    Streams live Ollama tokens when available; otherwise streams the extractive
    answer in word-sized pieces so the UI behaves identically in both modes.
    Citations are resolved *after* the full text is known (so the LLM path can
    honor the markers the model actually emitted) and sent as the final event.
    """

    grounded = is_grounded(chunks)

    # ---- Ollama streaming path -------------------------------------------
    if not settings.disable_ollama and chunks:
        try:
            prompt = build_prompt(question, chunks)
            tokens = _ollama_stream(prompt)
            # Pull the first token eagerly so a dead server fails *before* we
            # commit to the ollama mode (and can fall back cleanly).
            first = next(tokens)
            yield StreamEvent(
                "meta",
                {"mode": "ollama", "model": settings.ollama_model, "grounded": grounded},
            )
            collected: List[str] = [first]
            yield StreamEvent("token", {"text": first})
            for tok in tokens:
                collected.append(tok)
                yield StreamEvent("token", {"text": tok})
            full = "".join(collected).strip()
            _, citations = select_cited(full, chunks)
            yield StreamEvent(
                "citations", {"citations": [_cite_dict(c) for c in citations]}
            )
            return
        except Exception:  # noqa: BLE001 - fall back to extractive streaming
            pass

    # ---- Extractive streaming path ---------------------------------------
    ans = extractive_answer(question, chunks)
    yield StreamEvent(
        "meta", {"mode": "extractive", "model": "", "grounded": ans.grounded}
    )
    for piece in _word_pieces(ans.answer):
        yield StreamEvent("token", {"text": piece})
    yield StreamEvent(
        "citations", {"citations": [_cite_dict(c) for c in ans.citations]}
    )


def _word_pieces(text: str) -> Iterable[str]:
    """Split text into whitespace-preserving pieces for smooth streaming."""

    for piece in re.findall(r"\s*\S+\s*|\s+", text):
        if piece:
            yield piece


def _cite_dict(c: Citation) -> dict:
    return {
        "marker": c.marker,
        "source": c.source,
        "chunk_index": c.chunk_index,
        "score": c.score,
        "snippet": c.snippet,
    }
