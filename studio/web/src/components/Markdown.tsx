import { Children, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { studio } from "../lib/studio";

// A file path with a folder (src/phi/studio/worker.py:303, /Users/x/phi/robot-config.yaml), or one of
// the few machine files people name bare. WHY require a folder otherwise: "worker.py" alone could be
// several files, and a link that opens the wrong one is worse than no link.
const EXT = "py|tsx?|jsx?|json|ya?ml|md|sh|log|txt|toml|cfg|ini|css|html";
const PATH = new RegExp(
  `((?:~|\\.{1,2})?\\/?(?:[\\w.@-]+\\/)+[\\w.@-]+\\.(?:${EXT})|robot-config\\.yaml)(?::(\\d+))?`, "g",
);

function FileLink({ path, line, children }: { path: string; line: number | null; children: ReactNode }) {
  return (
    <button type="button" className="file-link" onClick={() => studio.openFile(path, line)}
      title={`Open ${path}${line ? ` at line ${line}` : ""} in Files`}>
      {children}
    </button>
  );
}

function linkify(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(PATH)) {
    const i = m.index ?? 0;
    if (i > last) out.push(text.slice(last, i));
    out.push(<FileLink key={i} path={m[1]} line={m[2] ? Number(m[2]) : null}>{m[0]}</FileLink>);
    last = i + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

const withLinks = (children: ReactNode): ReactNode =>
  Children.map(children, (c) => (typeof c === "string" ? linkify(c) : c));

const components: Components = {
  p: ({ children }) => <p>{withLinks(children)}</p>,
  li: ({ children }) => <li>{withLinks(children)}</li>,
  td: ({ children }) => <td>{withLinks(children)}</td>,
  a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>,
  pre: ({ children }) => <pre className="md-pre">{children}</pre>,
  code: ({ className, children }) => {
    const text = String(Children.toArray(children).join(""));
    const block = Boolean(className) || text.includes("\n");
    if (block) return <code className={className}>{children}</code>;
    const m = new RegExp(`^${PATH.source}$`).exec(text);
    if (m) return <FileLink path={m[1]} line={m[2] ? Number(m[2]) : null}><code>{text}</code></FileLink>;
    return <code>{children}</code>;
  },
};

// Claude's answers. react-markdown renders no raw HTML, so an answer that quotes a file cannot inject
// markup into Studio.
export function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>{text}</ReactMarkdown>
    </div>
  );
}
