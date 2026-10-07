import type { Citation } from "../api";

interface Props {
  citations: Citation[];
  turnId: number;
  // Marker currently highlighted (from a clicked [n] in the answer).
  highlighted?: number | null;
}

// Renders the sources an answer was grounded in, with score + snippet so the
// user can verify every claim against the original document. Each card has a
// stable id so a clicked [n] marker in the answer can scroll to and flash it.
export function CitationList({ citations, turnId, highlighted }: Props) {
  if (citations.length === 0) return null;
  return (
    <div className="citations">
      <div className="citations-title">Sources</div>
      <ol>
        {citations.map((c) => (
          <li
            key={c.marker}
            id={`cite-${turnId}-${c.marker}`}
            className={highlighted === c.marker ? "citation flash" : "citation"}
          >
            <div className="citation-head">
              <span className="citation-marker">[{c.marker}]</span>
              <span className="citation-source">{c.source}</span>
              <span className="citation-score">score {c.score.toFixed(3)}</span>
            </div>
            <p className="citation-snippet">{c.snippet}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
