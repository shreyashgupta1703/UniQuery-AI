import { useEffect, useMemo, useRef, useState } from "react";
import { Markdown } from "../components/Markdown";
import { answer, BrowserStore, chunkText, type Citation } from "./rag";
import { SAMPLE_DOCS, SAMPLE_QUESTIONS } from "./samples";
import { answerToMarkdown, downloadText } from "../export";

interface Turn {
  id: number;
  question: string;
  text: string;
  fullText: string;
  citations: Citation[];
  grounded: boolean;
  streaming: boolean;
}

// The flagship in-browser demo: the entire RAG pipeline (embed -> retrieve ->
// rerank -> cite -> extractive answer) runs client-side, with no backend and no
// network. It loads the sample corpus, lets the user ask questions, and renders
// ranked citations live — proving the "always works, fully offline" claim in a
// browser tab.
export default function Demo() {
  const store = useMemo(() => {
    const s = new BrowserStore();
    for (const doc of SAMPLE_DOCS) {
      // chunkText mirrors the backend paragraph-aware splitter.
      s.add(chunkText(doc.text, doc.name));
    }
    return s;
  }, []);

  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [source, setSource] = useState("");
  const [highlight, setHighlight] = useState<{ turn: number; marker: number } | null>(
    null,
  );
  const bottomRef = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);
  const sources = store.sources;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns.length]);

  function ask(q: string) {
    const question = q.trim();
    if (!question) return;
    const res = answer(store, question, 4, source || undefined);
    const id = nextId.current++;
    setInput("");
    setTurns((t) => [
      ...t,
      {
        id,
        question,
        text: "",
        fullText: res.answer,
        citations: res.citations,
        grounded: res.grounded,
        streaming: true,
      },
    ]);

    // Simulate token streaming so the demo feels like the real app.
    const pieces = res.answer.match(/\s*\S+\s*/g) ?? [res.answer];
    let i = 0;
    const tick = () => {
      i++;
      setTurns((t) =>
        t.map((turn) =>
          turn.id === id
            ? { ...turn, text: pieces.slice(0, i).join("") }
            : turn,
        ),
      );
      if (i < pieces.length) {
        window.setTimeout(tick, 18);
      } else {
        setTurns((t) =>
          t.map((turn) => (turn.id === id ? { ...turn, streaming: false } : turn)),
        );
      }
    };
    window.setTimeout(tick, 18);
  }

  function jump(turnId: number, marker: number) {
    const el = document.getElementById(`dcite-${turnId}-${marker}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      setHighlight({ turn: turnId, marker });
      window.setTimeout(() => setHighlight(null), 1600);
    }
  }

  function exportTurn(turn: Turn) {
    const md = answerToMarkdown(turn.question, turn.fullText, turn.citations, {
      mode: "extractive",
      model: "",
    });
    downloadText("localrag-demo-answer.md", md);
  }

  return (
    <div className="demo">
      <header className="demo-header">
        <div>
          <h1>localrag <span className="demo-tag">live demo</span></h1>
          <p className="demo-sub">
            The full retrieval pipeline runs <strong>entirely in your browser</strong> —
            no backend, no network, no API keys. Embeddings, cosine search,
            hybrid reranking, and cited extractive answers, all client-side.
          </p>
        </div>
        <a
          className="demo-repo"
          href="https://github.com/xj16/localrag"
          target="_blank"
          rel="noreferrer"
        >
          GitHub ↗
        </a>
      </header>

      <div className="demo-body">
        <aside className="demo-corpus">
          <h2>Corpus ({store.size} chunks)</h2>
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="">All sources</option>
            {sources.map((s) => (
              <option key={s.source} value={s.source}>
                {s.source}
              </option>
            ))}
          </select>
          <ul>
            {sources.map((s) => (
              <li key={s.source}>
                <span>{s.source}</span>
                <span className="chunk-count">{s.nChunks}</span>
              </li>
            ))}
          </ul>
          <p className="demo-note">
            Loaded from the repo's <code>sample_docs/</code>. Ask about solar
            power or coffee brewing.
          </p>
        </aside>

        <main className="demo-chat">
          <div className="demo-messages">
            {turns.length === 0 && (
              <div className="demo-welcome">
                <p>Try one of these:</p>
                <div className="demo-samples">
                  {SAMPLE_QUESTIONS.map((q) => (
                    <button key={q} onClick={() => ask(q)}>
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {turns.map((turn) => (
              <div key={turn.id} className="demo-turn">
                <div className="bubble user">{turn.question}</div>
                <div className="bubble bot">
                  <div className="answer-head">
                    <span className="answer-mode">extractive answer (in-browser)</span>
                    {!turn.streaming && !turn.grounded && (
                      <span className="low-confidence">low confidence</span>
                    )}
                    {!turn.streaming && (
                      <button className="export-btn" onClick={() => exportTurn(turn)}>
                        Export ↓
                      </button>
                    )}
                  </div>
                  <Markdown text={turn.text} onCite={(m) => jump(turn.id, m)} />
                  {turn.streaming && <span className="cursor" />}
                  {turn.citations.length > 0 && (
                    <div className="citations">
                      <div className="citations-title">Sources</div>
                      <ol>
                        {turn.citations.map((c) => (
                          <li
                            key={c.marker}
                            id={`dcite-${turn.id}-${c.marker}`}
                            className={
                              highlight?.turn === turn.id && highlight.marker === c.marker
                                ? "citation flash"
                                : "citation"
                            }
                          >
                            <div className="citation-head">
                              <span className="citation-marker">[{c.marker}]</span>
                              <span className="citation-source">{c.source}</span>
                              <span className="citation-score">
                                score {c.score.toFixed(3)}
                              </span>
                            </div>
                            <p className="citation-snippet">{c.snippet}</p>
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          <div className="composer">
            <div className="composer-row">
              <textarea
                className="composer-input"
                placeholder="Ask about solar power or coffee brewing…"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    ask(input);
                  }
                }}
                rows={2}
              />
              <button
                className="btn send"
                disabled={!input.trim()}
                onClick={() => ask(input)}
              >
                Ask
              </button>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
