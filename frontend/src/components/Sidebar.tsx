import { useRef, useState } from "react";
import type { Stats } from "../api";
import { deleteSource, ingestFile, ingestText } from "../api";

interface Props {
  stats: Stats | null;
  onChanged: () => void;
  theme: "dark" | "light";
  onToggleTheme: () => void;
}

// Left panel: shows backend status (embedder + Ollama), lists indexed sources,
// and provides both file-upload and paste-text ingestion.
export function Sidebar({ stats, onChanged, theme, onToggleTheme }: Props) {
  const [pasteText, setPasteText] = useState("");
  const [pasteName, setPasteName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      for (const file of Array.from(files)) {
        await ingestFile(file);
      }
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function handlePaste() {
    if (!pasteText.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await ingestText(pasteText, pasteName.trim() || "pasted.txt");
      setPasteText("");
      setPasteName("");
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(name: string) {
    setBusy(true);
    setError(null);
    try {
      await deleteSource(name);
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-row">
          <h1>localrag</h1>
          <button
            className="theme-toggle"
            onClick={onToggleTheme}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            aria-label="Toggle color theme"
          >
            {theme === "dark" ? "☀" : "☾"}
          </button>
        </div>
        <p className="tagline">offline document Q&amp;A + citations</p>
      </div>

      <section className="status">
        <StatusRow label="Embedder" value={stats?.embedder ?? "…"} />
        <StatusRow
          label="Generator"
          value={
            stats
              ? stats.ollama_available
                ? "Ollama (local LLM)"
                : "extractive fallback"
              : "…"
          }
          tone={stats?.ollama_available ? "ok" : "muted"}
        />
        <StatusRow label="Indexed chunks" value={String(stats?.n_chunks ?? 0)} />
      </section>

      <section className="ingest">
        <h2>Add documents</h2>
        <button
          className="btn"
          disabled={busy}
          onClick={() => fileRef.current?.click()}
        >
          Upload files (.md / .txt / .pdf)
        </button>
        <input
          ref={fileRef}
          type="file"
          multiple
          accept=".md,.markdown,.txt,.pdf"
          style={{ display: "none" }}
          onChange={(e) => handleFiles(e.target.files)}
        />

        <details className="paste">
          <summary>…or paste text</summary>
          <input
            className="input"
            placeholder="source name (e.g. notes.md)"
            value={pasteName}
            onChange={(e) => setPasteName(e.target.value)}
          />
          <textarea
            className="textarea"
            placeholder="Paste document text here"
            value={pasteText}
            onChange={(e) => setPasteText(e.target.value)}
          />
          <button className="btn" disabled={busy || !pasteText.trim()} onClick={handlePaste}>
            Ingest text
          </button>
        </details>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="sources">
        <h2>Sources ({stats?.sources.length ?? 0})</h2>
        {stats && stats.sources.length === 0 && (
          <p className="empty">No documents yet. Upload one to start.</p>
        )}
        <ul>
          {stats?.sources.map((s) => (
            <li key={s.source}>
              <span className="src-name" title={s.source}>
                {s.source}
              </span>
              <span className="src-chunks">{s.n_chunks}</span>
              <button
                className="src-del"
                title="remove"
                disabled={busy}
                onClick={() => handleDelete(s.source)}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      </section>
    </aside>
  );
}

function StatusRow({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "ok" | "muted";
}) {
  return (
    <div className="status-row">
      <span className="status-label">{label}</span>
      <span className={`status-value tone-${tone}`}>{value}</span>
    </div>
  );
}
