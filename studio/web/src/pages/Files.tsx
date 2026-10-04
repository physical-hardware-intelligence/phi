import {
  ArrowLeft, Check, Copy, FileText, LoaderCircle, MessageSquareText, RefreshCw, Search, Usb,
} from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import { studio, useStudio, type FileRef, type SearchHit, type SerialPort } from "../lib/studio";
import { Notices } from "../components/Notices";
import { label } from "../lib/labels";

// The files that answer "which port, which arm, which calibration" on this Mac, plus search over every
// folder Studio can show. Read-only: nothing on this page changes a file.
export function Files() {
  const link = useStudio((s) => s.link);
  const f = useStudio((s) => s.files);
  const [view, setView] = useState<"file" | "search">("file");

  useEffect(() => { if (link === "open") studio.loadFiles(); }, [link]);
  // A file opened from anywhere (an answer, a note, a search hit) takes the viewer.
  useEffect(() => { if (f.loading || f.open) setView("file"); }, [f.loading, f.open]);

  return (
    <div className="page">
      <Notices />
      <div className="files-grid">
        <div className="col">
          <SearchBox onSearch={() => setView("search")} />
          <PortsPanel />
          <NotesPanel />
          <CalibrationsPanel />
        </div>
        <section className="panel viewer">
          {view === "search" && (f.search || f.searching)
            ? <Results />
            : <Viewer onBack={f.search ? () => setView("search") : undefined} />}
        </section>
      </div>
    </div>
  );
}

// -- left column ------------------------------------------------------------------------------------

function SearchBox({ onSearch }: { onSearch: () => void }) {
  const [q, setQ] = useState("");
  const searching = useStudio((s) => s.files.searching);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (q.trim().length < 2) return;
    studio.search(q.trim());
    onSearch();
  };
  return (
    <form className="panel search-box" onSubmit={submit} role="search">
      <Search className="search-icon" aria-hidden />
      <input className="input" value={q} onChange={(e) => setQ(e.target.value)} maxLength={200}
        placeholder="Search files, e.g. usbmodem" aria-label="Search files" />
      <button className="btn btn-sm" disabled={q.trim().length < 2 || Boolean(searching)}>
        {searching ? <LoaderCircle className="spin" aria-hidden /> : null} Search
      </button>
    </form>
  );
}

function PortsPanel() {
  const ports = useStudio((s) => s.files.ports);
  const [all, setAll] = useState(false);
  const usb = ports?.ports.filter((p) => p.usb) ?? [];
  const system = (ports?.ports.length ?? 0) - usb.length;
  const shown = all ? ports?.ports ?? [] : usb;
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Serial ports</h2>
          <p className="panel-sub">{ports ? `Read ${clock(ports.at)}` : "Reading"}</p>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={() => studio.refreshPorts()}><RefreshCw aria-hidden /> Refresh</button>
      </div>
      {ports?.error ? (
        <div className="panel-body"><p className="text-warn">{ports.error}</p>{ports.fix && <p className="faint t-sm">{ports.fix}</p>}</div>
      ) : ports && shown.length === 0 ? (
        <p className="empty">No USB serial device is plugged in. Check each arm's USB cable and its power.</p>
      ) : (
        <ul className="port-list">{shown.map((p) => <PortRow key={p.device} p={p} />)}</ul>
      )}
      {system > 0 && (
        <div className="panel-foot">
          <button className="link-btn" onClick={() => setAll(!all)}>
            {all ? "Hide system ports" : `Show ${system} system ${system === 1 ? "port" : "ports"}`}
          </button>
        </div>
      )}
    </section>
  );
}

function PortRow({ p }: { p: SerialPort }) {
  const path = p.tty ?? p.device; // WHY the tty name: robot-config.yaml and LeRobot's --port use it
  const hex = (n: number | null) => (n === null ? "" : n.toString(16).padStart(4, "0"));
  return (
    <li className="port-row">
      <Usb className="port-icon" aria-hidden />
      <div className="port-text">
        <div className="port-main">
          <span className="mono ellipsis" title={path}>{path}</span>
          <CopyIcon text={path} label="Copy port" />
        </div>
        <div className="port-meta">
          {p.arm ? <span className="badge tone-ok">{label(p.arm)}</span> : p.usb && <span className="badge tone-neutral">No arm read</span>}
          {p.vid !== null && <span className="num">{hex(p.vid)}:{hex(p.pid)}</span>}
          {p.serial && <span className="num">SN {p.serial}</span>}
          {p.description && !p.arm && <span className="ellipsis">{p.description}</span>}
        </div>
      </div>
    </li>
  );
}

function NotesPanel() {
  const notes = useStudio((s) => s.files.index?.notes);
  const open = useStudio((s) => s.files.open);
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Rig notes</h2>
          <p className="panel-sub">Ports, arm ids and cameras for this Mac</p>
        </div>
      </div>
      {!notes ? <p className="empty">Reading</p> : notes.length === 0 ? (
        <p className="empty">No rig config found. It lives at robot-config.yaml in the main checkout.</p>
      ) : (
        <ul className="ref-list">
          {notes.map((n) => (
            <RefRow key={`${n.root}/${n.path}`} r={n} active={open?.root === n.root && open.path === n.path}
              title={n.label ?? n.path} sub={n.path} />
          ))}
        </ul>
      )}
    </section>
  );
}

function CalibrationsPanel() {
  const cals = useStudio((s) => s.files.index?.calibrations);
  const identity = useStudio((s) => s.identity);
  const open = useStudio((s) => s.files.open);
  const groups = useMemo(() => {
    const g = new Map<string, FileRef[]>();
    for (const c of cals ?? []) g.set(c.kind ?? "other", [...(g.get(c.kind ?? "other") ?? []), c]);
    return [...g.entries()];
  }, [cals]);
  const user = (id?: string) => identity.find((a) => a.expected === id)?.name;
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Calibration files</h2>
          <p className="panel-sub">LeRobot's calibration folder</p>
        </div>
      </div>
      {!cals ? <p className="empty">Reading</p> : cals.length === 0 ? (
        <p className="empty">No calibration files yet. Calibrate an arm to make one.</p>
      ) : groups.map(([kind, rows]) => (
        <div key={kind}>
          <div className="ref-group">{kind}</div>
          <ul className="ref-list">
            {rows.map((c) => (
              <RefRow key={c.path} r={c} active={open?.root === c.root && open.path === c.path} title={c.id ?? c.path}
                sub={user(c.id) ? `Used by ${user(c.id)}` : undefined} />
            ))}
          </ul>
        </div>
      ))}
    </section>
  );
}

function RefRow({ r, title, sub, active }: { r: FileRef; title: string; sub?: string; active: boolean }) {
  return (
    <li>
      <button className={`ref-row ${active ? "is-active" : ""}`} onClick={() => studio.openFile(r.path, null, r.root)}>
        <FileText aria-hidden />
        <span className="ref-text">
          <span className="ref-title">{title}</span>
          {sub && <span className="ref-sub mono">{sub}</span>}
        </span>
        <span className="ref-time faint num">{ago(r.mtime)}</span>
      </button>
    </li>
  );
}

// -- right column -----------------------------------------------------------------------------------

function Viewer({ onBack }: { onBack?: () => void }) {
  const f = useStudio((s) => s.files);
  const roots = useStudio((s) => s.files.index?.roots);
  const body = useRef<HTMLDivElement>(null);
  const file = f.open;
  const lines = useMemo(() => file?.text.split("\n") ?? [], [file]);

  // WHY scroll the panel, not the page: the target line should land in view without moving the left column.
  useLayoutEffect(() => {
    const el = body.current;
    if (!el || !file) return;
    el.scrollTop = 0;
    const row = file.line ? el.querySelector<HTMLElement>(`[data-line="${file.line}"]`) : null;
    if (row) el.scrollTop = Math.max(0, row.getBoundingClientRect().top - el.getBoundingClientRect().top - el.clientHeight / 3);
  }, [file]);

  if (f.loading) {
    return <Placeholder><LoaderCircle className="spin" aria-hidden /> Opening {f.loading}</Placeholder>;
  }
  if (f.error && !f.searching) {
    return (
      <Placeholder>
        <div className="viewer-error">
          <p className="strong">{f.error.message}</p>
          <p className="faint t-sm">Studio shows files in its own folders only, and never files that may hold secrets.</p>
          {onBack && <button className="btn btn-sm" onClick={onBack}><ArrowLeft aria-hidden /> Back to results</button>}
        </div>
      </Placeholder>
    );
  }
  if (!file) {
    return (
      <Placeholder>
        <div className="viewer-error">
          <p className="strong">Open a file</p>
          <p className="faint t-sm">Pick a rig note or calibration file, search, or click a file path in a Claude answer.</p>
        </div>
      </Placeholder>
    );
  }

  const label = roots?.find((r) => r.key === file.root)?.label ?? file.root;
  const where = file.line ? `${file.abs}:${file.line}` : file.abs;
  return (
    <>
      <div className="viewer-head">
        {onBack && (
          <button className="btn btn-ghost btn-sm btn-icon" onClick={onBack} aria-label="Back to results" title="Back to results">
            <ArrowLeft aria-hidden />
          </button>
        )}
        <div className="viewer-title">
          <span className="mono ellipsis" title={file.abs}>{file.path}</span>
          <span className="faint t-cap">{label}, {size(file.size)}, changed {ago(file.mtime)}</span>
        </div>
        <CopyIcon text={where} label="Copy full path" withText />
        <button className="btn btn-sm" onClick={() => studio.openAssistant({ message: `The file ${where}` })}>
          <MessageSquareText aria-hidden /> Ask Claude
        </button>
      </div>
      {file.truncated && <div className="viewer-note">Showing the first 512 KB of {size(file.size)}.</div>}
      <div className="code" ref={body} role="region" aria-label={file.path} tabIndex={0}>
        {lines.map((text, i) => (
          <div key={i} data-line={i + 1} className={`code-line ${file.line === i + 1 ? "is-target" : ""}`}>
            <span className="code-n num" aria-hidden>{i + 1}</span>
            <span className="code-t">{text || " "}</span>
          </div>
        ))}
      </div>
    </>
  );
}

function Results() {
  const search = useStudio((s) => s.files.search);
  const searching = useStudio((s) => s.files.searching);
  const error = useStudio((s) => s.files.error);
  const byFile = useMemo(() => {
    const g = new Map<string, SearchHit[]>();
    for (const h of search?.hits ?? []) {
      const k = `${h.root}\u0000${h.path}`;
      g.set(k, [...(g.get(k) ?? []), h]);
    }
    return [...g.values()];
  }, [search]);

  if (searching) return <Placeholder><LoaderCircle className="spin" aria-hidden /> Searching for “{searching}”</Placeholder>;
  if (!search) return <Placeholder>{error?.message ?? "Search the folders Studio can show."}</Placeholder>;
  const n = search.hits.length;
  return (
    <>
      <div className="viewer-head">
        <div className="viewer-title">
          <span className="strong">{n === 0 ? "No matches" : `${n} ${n === 1 ? "match" : "matches"}`} for “{search.query}”</span>
          <span className="faint t-cap num">
            {byFile.length} {byFile.length === 1 ? "file" : "files"} of {search.scanned} searched
            {search.stopped ? ". Stopped early: refine the search to see the rest" : ""}
          </span>
        </div>
      </div>
      <div className="code results">
        {byFile.map((hits) => (
          <div key={`${hits[0].root}/${hits[0].path}`} className="hit-file">
            <div className="hit-path mono">{hits[0].path}</div>
            {hits.map((h) => (
              <button key={h.line} className="hit" onClick={() => studio.openFile(h.path, h.line, h.root)}>
                <span className="code-n num">{h.line}</span>
                <span className="code-t">{mark(h.text, search.query)}</span>
              </button>
            ))}
          </div>
        ))}
      </div>
    </>
  );
}

// -- small parts ------------------------------------------------------------------------------------

function Placeholder({ children }: { children: ReactNode }) {
  return <div className="viewer-empty">{children}</div>;
}

function CopyIcon({ text, label, withText }: { text: string; label: string; withText?: boolean }) {
  const [done, setDone] = useState(false);
  const copy = () => { void navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); };
  return (
    <button className={`btn btn-ghost btn-sm ${withText ? "" : "btn-icon"}`} onClick={copy} aria-label={label} title={label}>
      {done ? <Check aria-hidden /> : <Copy aria-hidden />}{withText && (done ? "Copied" : "Copy path")}
    </button>
  );
}

function mark(text: string, query: string): ReactNode {
  const i = text.toLowerCase().indexOf(query.toLowerCase());
  if (i < 0) return text;
  return <>{text.slice(0, i)}<mark>{text.slice(i, i + query.length)}</mark>{text.slice(i + query.length)}</>;
}

function ago(mtimeS: number): string {
  const s = Math.max(0, Date.now() / 1000 - mtimeS);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  const d = Math.floor(s / 86400);
  return d < 30 ? `${d} d ago` : new Date(mtimeS * 1000).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
}

function size(b: number): string {
  return b < 1024 ? `${b} B` : b < 1024 * 1024 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1024 / 1024).toFixed(1)} MB`;
}

function clock(ms: number): string {
  return new Date(ms).toLocaleTimeString([], { hour12: false });
}
