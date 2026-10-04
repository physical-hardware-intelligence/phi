import {
  CircleCheck, CircleDashed, CircleX, CornerDownRight, FileText, Info, LoaderCircle, MessageSquareText, Play, TriangleAlert,
} from "lucide-react";
import { useEffect, useState, type ComponentType } from "react";
import { studio, useStudio, type CheckResult, type CheckStatus } from "../lib/studio";

const GROUPS: Array<{ key: CheckResult["group"]; sub: string }> = [
  { key: "This Mac", sub: "Software, rig config, ports, calibration files, disk" },
  { key: "Rig", sub: "Live from the servos and cameras" },
  { key: "Studio", sub: "The assistant, the interface build, the running code" },
];

type Icon = ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
const LOOK: Record<CheckStatus, { icon: Icon; tone: string; label: string; many: string }> = {
  fail: { icon: CircleX, tone: "danger", label: "Failed", many: "failed" },
  warn: { icon: TriangleAlert, tone: "warn", label: "Warning", many: "warnings" },
  pass: { icon: CircleCheck, tone: "ok", label: "Passed", many: "passed" },
  info: { icon: Info, tone: "neutral", label: "Info", many: "info" },
  skip: { icon: CircleDashed, tone: "neutral", label: "Skipped", many: "skipped" },
};
const ORDER: CheckStatus[] = ["fail", "warn", "pass", "info", "skip"];

// Every read-only check Studio can make, in one run. Nothing here moves an arm.
export function Checks() {
  const c = useStudio((s) => s.checks);
  const link = useStudio((s) => s.link);
  const [hidePassed, setHidePassed] = useState(false);

  // First visit runs them; later visits show the last run until asked again.
  useEffect(() => { if (link === "open" && !c.results && !c.running) studio.runChecks(); }, [link]); // eslint-disable-line react-hooks/exhaustive-deps

  const results = c.results ?? [];
  const count = (st: CheckStatus) => results.filter((r) => r.status === st).length;
  const bad = count("fail") + count("warn");
  return (
    <div className="page">
      <section className="panel checks-head">
        <div className="checks-verdict">
          {!c.results ? (
            <span className="checks-title">{c.running ? "Running checks" : "Not run yet"}</span>
          ) : (
            <span className={`checks-title tone-${count("fail") ? "danger" : count("warn") ? "warn" : "ok"}`}>
              {[count("fail") && `${count("fail")} failed`, count("warn") && warnings(count("warn"))].filter(Boolean).join(", ")
                || "Everything Studio can check passed"}
            </span>
          )}
          <span className="checks-sub num">
            {c.results && c.at
              ? `${results.length} checks, ran ${new Date(c.at).toLocaleTimeString([], { hour12: false })} in ${c.ms} ms. Read-only: nothing moves.`
              : "Read-only: nothing moves and no register is written."}
          </span>
        </div>
        {c.results && (
          <ul className="tally" aria-label="Results by status">
            {ORDER.map((st) => {
              const { icon: Icon, tone, label, many } = LOOK[st];
              return (
                <li key={st} className={`tone-${tone} ${count(st) ? "" : "is-zero"}`}>
                  <Icon aria-hidden /><span className="num">{count(st)}</span> {count(st) === 1 ? label.toLowerCase() : many}
                </li>
              );
            })}
          </ul>
        )}
        <div className="checks-actions">
          <label className="switch">
            <input type="checkbox" checked={hidePassed} onChange={(e) => setHidePassed(e.target.checked)} />
            <span>Only problems</span>
          </label>
          <button className="btn btn-primary" onClick={() => studio.runChecks()} disabled={c.running || link !== "open"}>
            {c.running ? <LoaderCircle className="spin" aria-hidden /> : <Play aria-hidden />}
            {c.results ? "Run again" : "Run checks"}
          </button>
        </div>
      </section>

      {hidePassed && c.results && bad === 0 && <p className="empty panel">No failures and no warnings.</p>}

      {GROUPS.map((g) => {
        const rows = results.filter((r) => r.group === g.key && (!hidePassed || r.status === "fail" || r.status === "warn"));
        if (!c.results || rows.length === 0) return null;
        return (
          <section key={g.key} className={`panel ${c.running ? "is-stale" : ""}`}>
            <div className="panel-head">
              <div>
                <h2 className="panel-title">{g.key}</h2>
                <p className="panel-sub">{g.sub}</p>
              </div>
            </div>
            <ul className="check-list">
              {rows.filter((r) => r.status !== "skip").map((r) => <CheckRow key={r.id} r={r} />)}
              {skipped(rows).map(([why, titles]) => (
                <li key={why} className="check-row is-skip tone-neutral">
                  <CircleDashed className="check-icon" aria-hidden />
                  <div className="check-text">
                    <div className="check-title">{titles.length === 1 ? titles[0] : why}</div>
                    <div className="check-detail">{titles.length === 1 ? why : `Skipped ${titles.length} checks: ${titles.join(", ")}`}</div>
                  </div>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}

const warnings = (n: number) => `${n} ${n === 1 ? "warning" : "warnings"}`;

/** Skipped checks that share a reason, as [reason, titles], so a rig that is off costs one line, not nine. */
function skipped(rows: CheckResult[]): Array<[string, string[]]> {
  const g = new Map<string, string[]>();
  for (const r of rows) if (r.status === "skip") g.set(r.detail, [...(g.get(r.detail) ?? []), r.title]);
  return [...g.entries()];
}

function CheckRow({ r }: { r: CheckResult }) {
  const { icon: Icon, tone, label } = LOOK[r.status];
  const problem = r.status === "fail" || r.status === "warn";
  return (
    <li className={`check-row tone-${tone} is-${r.status}`}>
      <Icon className="check-icon" aria-hidden />
      <div className="check-text">
        <div className="check-title">{r.title}<span className="sr-only">: {label}</span></div>
        <div className="check-detail">{r.detail}</div>
        {problem && r.fix && (
          <div className="check-fix"><CornerDownRight aria-hidden /><span>{r.fix}</span></div>
        )}
      </div>
      <div className="check-actions">
        {r.file && (
          <button className="btn btn-ghost btn-sm" onClick={() => studio.openFile(r.file!.path, r.file!.line, r.file!.root)}
            title={`Open ${r.file.path}${r.file.line ? ` at line ${r.file.line}` : ""}`}>
            <FileText aria-hidden /> {r.file.line ? `Line ${r.file.line}` : "Open"}
          </button>
        )}
        {problem && (
          <button className="btn btn-sm" onClick={() => studio.openAssistant({ message: `Check “${r.title}” ${r.status === "fail" ? "failed" : "warned"}: ${r.detail}`, fix: r.fix ?? undefined })}>
            <MessageSquareText aria-hidden /> Ask Claude
          </button>
        )}
      </div>
    </li>
  );
}
