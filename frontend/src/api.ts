// Typed client for the localrag FastAPI backend.

export interface Citation {
  marker: number;
  source: string;
  chunk_index: number;
  score: number;
  snippet: string;
}

export interface AskResponse {
  answer: string;
  mode: string;
  model: string;
  grounded: boolean;
  citations: Citation[];
}

// Streaming event shapes emitted by POST /api/ask/stream (SSE).
export type StreamEvent =
  | { type: "meta"; mode: string; model: string; grounded: boolean }
  | { type: "token"; text: string }
  | { type: "citations"; citations: Citation[] }
  | { type: "done" };

export interface IngestResponse {
  source: string;
  n_chunks: number;
  document_id: number;
}

export interface SourceInfo {
  source: string;
  n_chunks: number;
  created_at?: string | null;
}

export interface Stats {
  embedder: string;
  embed_dim: number | null;
  n_chunks: number;
  ollama_available: boolean;
  sources: SourceInfo[];
}

// Allow overriding the API base (e.g. when the SPA is hosted separately).
const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore parse errors */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function getStats(): Promise<Stats> {
  return handle<Stats>(await fetch(`${BASE}/api/stats`));
}

export async function ask(
  question: string,
  topK?: number,
  source?: string,
): Promise<AskResponse> {
  const res = await fetch(`${BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK, source: source ?? null }),
  });
  return handle<AskResponse>(res);
}

// Stream an answer as it is generated. Parses the SSE `data:` frames from
// /api/ask/stream and invokes `onEvent` for each. Falls back cleanly if the
// browser/environment doesn't support streaming bodies.
export async function askStream(
  question: string,
  onEvent: (ev: StreamEvent) => void,
  opts: { topK?: number; source?: string; signal?: AbortSignal } = {},
): Promise<void> {
  const res = await fetch(`${BASE}/api/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      top_k: opts.topK,
      source: opts.source ?? null,
    }),
    signal: opts.signal,
  });
  if (!res.ok || !res.body) {
    // Surface a useful error using the same envelope as the JSON endpoints.
    await handle<unknown>(res);
    throw new Error("stream unavailable");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const flush = (chunk: string) => {
    buffer += chunk;
    let idx: number;
    // SSE frames are separated by a blank line.
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const json = line.slice(5).trim();
        if (!json) continue;
        try {
          onEvent(JSON.parse(json) as StreamEvent);
        } catch {
          /* ignore malformed frame */
        }
      }
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    flush(decoder.decode(value, { stream: true }));
  }
  flush(decoder.decode());
}

export async function ingestText(
  text: string,
  source: string,
): Promise<IngestResponse> {
  const res = await fetch(`${BASE}/api/ingest/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source }),
  });
  return handle<IngestResponse>(res);
}

export async function ingestFile(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/ingest/file`, {
    method: "POST",
    body: form,
  });
  return handle<IngestResponse>(res);
}

export async function deleteSource(name: string): Promise<void> {
  const res = await fetch(`${BASE}/api/sources/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  await handle<unknown>(res);
}
