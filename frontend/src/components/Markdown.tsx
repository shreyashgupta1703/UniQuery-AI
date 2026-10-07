import type { ReactNode } from "react";

// A tiny, dependency-free Markdown renderer scoped to what an LLM answer
// actually produces: paragraphs, bold/italic, inline code, fenced code blocks,
// unordered/ordered lists, and — crucially — clickable [n] citation markers.
//
// Every string goes through React's own escaping (we build elements, never set
// innerHTML), so this is XSS-safe by construction: raw HTML in the answer is
// rendered as literal text, not executed.

interface Props {
  text: string;
  // Called when a [n] citation marker is clicked, with the 1-based marker.
  onCite?: (marker: number) => void;
}

const CITE_RE = /(\[\d+\])/g;
const INLINE_RE = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[\d+\])/g;

function renderInline(text: string, onCite?: (m: number) => void): ReactNode[] {
  const out: ReactNode[] = [];
  const parts = text.split(INLINE_RE);
  parts.forEach((part, i) => {
    if (!part) return;
    if (/^\[\d+\]$/.test(part)) {
      const marker = parseInt(part.slice(1, -1), 10);
      out.push(
        <button
          key={i}
          type="button"
          className="marker"
          onClick={() => onCite?.(marker)}
          title={`Jump to source [${marker}]`}
        >
          {part}
        </button>,
      );
    } else if (part.startsWith("**") && part.endsWith("**")) {
      out.push(<strong key={i}>{part.slice(2, -2)}</strong>);
    } else if (part.startsWith("`") && part.endsWith("`")) {
      out.push(<code key={i}>{part.slice(1, -1)}</code>);
    } else if (part.startsWith("*") && part.endsWith("*")) {
      out.push(<em key={i}>{part.slice(1, -1)}</em>);
    } else {
      out.push(<span key={i}>{part}</span>);
    }
  });
  return out;
}

export function Markdown({ text, onCite }: Props) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block.
    if (line.trim().startsWith("```")) {
      const code: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        code.push(lines[i]);
        i++;
      }
      i++; // closing fence
      blocks.push(
        <pre key={key++} className="md-code">
          <code>{code.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    // Headings.
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) {
      const level = h[1].length;
      const Tag = (`h${Math.min(level + 2, 6)}`) as "h3" | "h4" | "h5" | "h6";
      blocks.push(<Tag key={key++}>{renderInline(h[2], onCite)}</Tag>);
      i++;
      continue;
    }

    // Unordered list.
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ""));
        i++;
      }
      blocks.push(
        <ul key={key++} className="md-list">
          {items.map((it, j) => (
            <li key={j}>{renderInline(it, onCite)}</li>
          ))}
        </ul>,
      );
      continue;
    }

    // Ordered list.
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i++;
      }
      blocks.push(
        <ol key={key++} className="md-list">
          {items.map((it, j) => (
            <li key={j}>{renderInline(it, onCite)}</li>
          ))}
        </ol>,
      );
      continue;
    }

    // Blank line -> paragraph break.
    if (!line.trim()) {
      i++;
      continue;
    }

    // Paragraph: gather consecutive non-blank, non-special lines.
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].trim().startsWith("```") &&
      !/^#{1,4}\s+/.test(lines[i]) &&
      !/^\s*[-*]\s+/.test(lines[i]) &&
      !/^\s*\d+\.\s+/.test(lines[i])
    ) {
      para.push(lines[i]);
      i++;
    }
    blocks.push(
      <p key={key++} className="md-p">
        {renderInline(para.join(" "), onCite)}
      </p>,
    );
  }

  return <div className="markdown">{blocks}</div>;
}

// Re-export the marker regex so callers can detect whether an answer cites.
export { CITE_RE };
