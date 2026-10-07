// A faithful, dependency-free TypeScript port of localrag's *fallback* pipeline:
// the hashing embedder, cosine similarity, paragraph-aware chunking, lexical
// hybrid reranking, honest citations, and the extractive answerer. It mirrors
// backend/app/{embeddings,chunking,store,generation}.py closely enough that the
// browser demo behaves like the real offline path — no server, no network, no
// WASM runtime, no numpy. Everything runs client-side.
//
// Note on hashing: the Python side uses BLAKE2b for feature hashing. Here we use
// a self-consistent FNV-1a hash instead of pulling in a crypto dependency. Since
// the demo embeds both the corpus and the query with *this same* embedder, the
// retrieval geometry is internally consistent — which is all that matters for a
// client-side demonstration. The algorithm (unigrams + bigrams, signed feature
// hashing, L2 normalization, cosine) is identical.

export interface Chunk {
  text: string;
  source: string;
  chunkIndex: number;
}

export interface ScoredChunk extends Chunk {
  score: number;
}

export interface Citation {
  marker: number;
  source: string;
  chunkIndex: number;
  score: number;
  snippet: string;
}

export interface Answer {
  answer: string;
  citations: Citation[];
  grounded: boolean;
}

const HASH_DIM = 512;
const GROUNDING_THRESHOLD = 0.15;

// -------------------------------------------------------------- tokenization
function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9]+/g) ?? []);
}

function ngrams(tokens: string[], n: number): string[] {
  if (n <= 1) return tokens;
  const out: string[] = [];
  for (let i = 0; i + n <= tokens.length; i++) {
    out.push(tokens.slice(i, i + n).join(" "));
  }
  return out;
}

// FNV-1a 32-bit hash -> (bucket, sign). Deterministic and dependency-free.
function hashFeature(feature: string): [number, number] {
  let h = 0x811c9dc5;
  for (let i = 0; i < feature.length; i++) {
    h ^= feature.charCodeAt(i);
    // multiply by the FNV prime (16777619) with 32-bit overflow
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  const bucket = h % HASH_DIM;
  const sign = (h & 0x80000000) !== 0 ? 1 : -1;
  return [bucket, sign];
}

function l2normalize(vec: Float64Array): Float64Array {
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm === 0) return vec;
  for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

export function embed(text: string): Float64Array {
  const vec = new Float64Array(HASH_DIM);
  const tokens = tokenize(text);
  if (tokens.length === 0) return vec;
  const features = ngrams(tokens, 1).concat(ngrams(tokens, 2));
  for (const feat of features) {
    const [bucket, sign] = hashFeature(feat);
    vec[bucket] += sign;
  }
  return l2normalize(vec);
}

export function cosine(a: Float64Array, b: Float64Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // both inputs are L2-normalized
}

// ------------------------------------------------------------------ chunking
// Paragraph-aware windows with overlap, mirroring chunk_text() in Python.
export function chunkText(
  text: string,
  source: string,
  chunkSize = 700,
  chunkOverlap = 120,
): Chunk[] {
  const normalized = text.replace(/\r\n/g, "\n").replace(/[ \t]+/g, " ").trim();
  if (!normalized) return [];
  const units = normalized
    .split(/\n\s*\n/)
    .map((u) => u.trim())
    .filter(Boolean);

  const expanded: string[] = [];
  for (const unit of units) {
    if (unit.length <= chunkSize) {
      expanded.push(unit);
    } else {
      for (let i = 0; i < unit.length; i += chunkSize) {
        expanded.push(unit.slice(i, i + chunkSize));
      }
    }
  }

  const chunks: Chunk[] = [];
  let buffer = "";
  for (const unit of expanded) {
    const candidate = buffer ? `${buffer}\n\n${unit}` : unit;
    if (candidate.length <= chunkSize || !buffer) {
      buffer = candidate;
    } else {
      chunks.push({ text: buffer.trim(), source, chunkIndex: chunks.length });
      const tail = chunkOverlap ? buffer.slice(-chunkOverlap) : "";
      buffer = tail ? `${tail}\n\n${unit}` : unit;
    }
  }
  if (buffer.trim()) {
    chunks.push({ text: buffer.trim(), source, chunkIndex: chunks.length });
  }
  return chunks;
}

// -------------------------------------------------------------------- store
export class BrowserStore {
  private chunks: Chunk[] = [];
  private vectors: Float64Array[] = [];

  add(chunks: Chunk[]): void {
    for (const c of chunks) {
      this.chunks.push(c);
      this.vectors.push(embed(c.text));
    }
  }

  clear(): void {
    this.chunks = [];
    this.vectors = [];
  }

  get sources(): { source: string; nChunks: number }[] {
    const counts = new Map<string, number>();
    for (const c of this.chunks) {
      counts.set(c.source, (counts.get(c.source) ?? 0) + 1);
    }
    return [...counts.entries()].map(([source, nChunks]) => ({ source, nChunks }));
  }

  get size(): number {
    return this.chunks.length;
  }

  search(query: string, topK: number, source?: string): ScoredChunk[] {
    const q = embed(query);
    const scored: ScoredChunk[] = [];
    for (let i = 0; i < this.chunks.length; i++) {
      if (source && this.chunks[i].source !== source) continue;
      scored.push({ ...this.chunks[i], score: cosine(q, this.vectors[i]) });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }
}

// ---------------------------------------------------------------- reranking
export function lexicalRerank(
  question: string,
  chunks: ScoredChunk[],
  topK: number,
  weight = 0.35,
): ScoredChunk[] {
  const qTerms = new Set(tokenize(question));
  if (qTerms.size === 0) return chunks.slice(0, topK);
  const reranked = chunks.map((ch) => {
    const cTerms = new Set(tokenize(ch.text));
    let overlap = 0;
    for (const t of qTerms) if (cTerms.has(t)) overlap++;
    const blended = (1 - weight) * ch.score + weight * (overlap / qTerms.size);
    return { ...ch, score: blended };
  });
  reranked.sort((a, b) => b.score - a.score);
  return reranked.slice(0, topK);
}

// ------------------------------------------------------------------ answering
function snippet(text: string, limit = 240): string {
  const s = text.trim().replace(/\s+/g, " ");
  return s.length > limit ? s.slice(0, limit - 3).trimEnd() + "..." : s;
}

function topSentences(text: string, query: string, max = 2): string[] {
  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (sentences.length === 0) return [];
  const qTerms = new Set(tokenize(query));
  if (qTerms.size === 0) return sentences.slice(0, max);
  const ranked = sentences
    .map((s, i) => {
      const sTerms = new Set(tokenize(s));
      let overlap = 0;
      for (const t of qTerms) if (sTerms.has(t)) overlap++;
      return { i, overlap };
    })
    .sort((a, b) => b.overlap - a.overlap)
    .slice(0, max)
    .map((r) => r.i)
    .sort((a, b) => a - b);
  return ranked.map((i) => sentences[i]);
}

export function isGrounded(chunks: ScoredChunk[]): boolean {
  return chunks.some((c) => c.score >= GROUNDING_THRESHOLD);
}

export function extractiveAnswer(question: string, chunks: ScoredChunk[]): Answer {
  if (chunks.length === 0) {
    return {
      answer:
        "I don't have any indexed documents that match this question. Try adding a relevant document first.",
      citations: [],
      grounded: false,
    };
  }
  const citations: Citation[] = chunks.map((ch, i) => ({
    marker: i + 1,
    source: ch.source,
    chunkIndex: ch.chunkIndex,
    score: Math.round(ch.score * 10000) / 10000,
    snippet: snippet(ch.text),
  }));

  const lines: string[] = [];
  chunks.forEach((ch, i) => {
    const sents = topSentences(ch.text, question, 2);
    if (sents.length) lines.push(`${sents.join(" ")} [${i + 1}]`);
  });
  const body = lines.length ? lines.join(" ") : "See the cited passages below.";
  const preamble =
    "Based on the most relevant passages in the indexed documents (this in-browser demo runs the extractive path — no LLM, no server):\n\n";
  return { answer: preamble + body, citations, grounded: isGrounded(chunks) };
}

// One-call pipeline used by the demo UI.
export function answer(
  store: BrowserStore,
  question: string,
  topK = 4,
  source?: string,
): Answer {
  const candidates = store.search(question, topK * 3, source);
  const reranked = lexicalRerank(question, candidates, topK);
  return extractiveAnswer(question, reranked);
}
