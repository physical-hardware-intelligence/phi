import * as Dialog from "@radix-ui/react-dialog";
import { Check, Flag, Undo2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { ActionBar } from "../components/ActionBar";
import { Cameras } from "../components/Cameras";
import { Notices } from "../components/Notices";
import { pct, studio, useStudio, wilson, type EvalRecord, type EvalSummary } from "../lib/studio";
import { PolicySetup, usePolicyChoice } from "./Policy";

// An eval is N episodes of one policy on one task, each judged by a person. The server keeps the record and
// saves it after every judgement; this page runs the loop: run an episode, judge it, reset the scene, repeat.
export function Evaluate() {
  const evals = useStudio((s) => s.evals);
  if (!evals) {
    return (
      <div className="page">
        <Notices />
        <section className="panel"><p className="empty">Evals need a data directory. Start Studio with <span className="mono">--data-dir</span>.</p></section>
      </div>
    );
  }
  return evals.current ? <Running rec={evals.current} /> : <Setup past={evals.past} dir={evals.dir} />;
}

// -- before: choose what to evaluate ----------------------------------------------------------------
function Setup({ past, dir }: { past: EvalSummary[]; dir: string }) {
  const policies = useStudio((s) => s.rig?.policies ?? []);
  const control = useStudio((s) => s.control);
  const choice = usePolicyChoice(policies);
  const [planned, setPlanned] = useState(10);
  const okN = Number.isInteger(planned) && planned >= 1 && planned <= 500;
  const [lo, hi] = wilson(Math.round(planned * 0.8), okN ? planned : 1);
  const why = !control ? "Another window has control" : choice.problem ?? (okN ? null : "Plan 1 to 500 episodes");

  return (
    <div className="page evaluate">
      <Notices />
      <div className="eval-setup">
        <PolicySetup policies={policies} choice={choice} title="New eval">
          <label className="field">
            <span className="field-label">Episodes</span>
            <input className="input num" type="number" min={1} max={500} value={planned}
              onChange={(e) => setPlanned(Number(e.target.value))} />
            {okN && (
              <span className="field-hint">
                If {Math.round(planned * 0.8)} of {planned} succeed, the 95% interval is {pct(lo)} to {pct(hi)}.
                More episodes narrow it.
              </span>
            )}
          </label>
          <div className="form-actions">
            {why && <span className="field-hint">{why}</span>}
            <button className="btn btn-primary" disabled={why !== null}
              onClick={() => studio.send({ cmd: "eval_begin", policy: choice.policy, task: choice.task, planned, limit_s: choice.limit })}>
              <Flag aria-hidden /> Start eval
            </button>
          </div>
        </PolicySetup>
        <section className="panel">
          <div className="panel-head">
            <div>
              <h2 className="panel-title">Past evals</h2>
              <p className="panel-sub">Saved in <span className="mono">{dir}</span></p>
            </div>
          </div>
          {past.length === 0 ? <p className="empty">No evals yet.</p> : (
            <div className="table-scroll">
              <table className="table past">
                <thead><tr><th>Started</th><th>Policy</th><th>Task</th><th className="r">Episodes</th><th>Success, 95% interval</th></tr></thead>
                <tbody>
                  {past.map((e) => (
                    <tr key={e.id}>
                      <td className="num">{when(e.started_at)}</td>
                      <td>{e.policy}</td>
                      <td className="ellipsis" title={e.task}>{e.task || <span className="faint">none</span>}</td>
                      <td className="r num">{e.n}<span className="faint"> / {e.planned}</span></td>
                      <td><Interval k={e.successes} n={e.n} lo={e.ci95[0]} hi={e.ci95[1]} compact /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

const when = (t: number) => new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });

// -- during: run, judge, reset, repeat -------------------------------------------------------------------
function Running({ rec }: { rec: EvalRecord }) {
  const run = useStudio((s) => s.telemetry?.policy ?? null);
  // Only a run with this eval's policy, task and time limit, started after the eval began, is an episode.
  const mine = run !== null && run.id === rec.policy && run.task === rec.task && run.limit_s === rec.limit_s
    && run.started_at >= rec.started_at;
  const judged = mine && rec.episodes.some((e) => e.run_id === run.run_id);
  const toJudge = mine && !run.running && !judged;
  const next = rec.n + 1;

  return (
    <div className="page evaluate">
      <Summary rec={rec} />
      <ActionBar activity="policy" startLabel={`Run episode ${next}`} canStart={!toJudge}
        startBlocked={toJudge ? "Judge the last episode first" : undefined}
        startMsg={{ policy: rec.policy, task: rec.task, limit_s: rec.limit_s }} />
      <Notices />
      <div className="work-grid">
        <div className="col">
          <Judge rec={rec} toJudge={toJudge} judged={judged} running={mine && run.running} />
          <Cameras />
        </div>
        <div className="col rail">
          <Episodes rec={rec} />
        </div>
      </div>
    </div>
  );
}

function Summary({ rec }: { rec: EvalRecord }) {
  const control = useStudio((s) => s.control);
  const [confirm, setConfirm] = useState(false);
  const end = () => (rec.n < rec.planned ? setConfirm(true) : studio.send({ cmd: "eval_end" }));
  return (
    <section className="panel eval-summary">
      <div className="eval-what">
        <span className="t-cap faint">Evaluating</span>
        <span className="eval-policy">{rec.policy}</span>
        <span className="muted ellipsis" title={rec.task}>{rec.task || "No task text"}</span>
      </div>
      <div className="eval-stat">
        <span className="t-cap faint">Judged</span>
        <span className="t-stat num">{rec.n}<span className="eval-of">{rec.n > rec.planned ? ` (${rec.planned} planned)` : ` / ${rec.planned}`}</span></span>
      </div>
      <div className="eval-stat eval-rate">
        <span className="t-cap faint">Success rate, 95% interval</span>
        <Interval k={rec.successes} n={rec.n} lo={rec.ci95[0]} hi={rec.ci95[1]} />
      </div>
      <div className="eval-end">
        <button className={`btn ${rec.n >= rec.planned ? "btn-primary" : ""}`} onClick={end} disabled={!control}>End eval</button>
      </div>
      <Dialog.Root open={confirm} onOpenChange={setConfirm}>
        <Dialog.Portal>
          <Dialog.Overlay className="dialog-overlay" />
          <Dialog.Content className="dialog" aria-describedby="end-eval-desc">
            <Dialog.Title className="dialog-title">End the eval at {rec.n} of {rec.planned} episodes?</Dialog.Title>
            <p id="end-eval-desc" className="dialog-text">
              The record keeps the {rec.n} judged episodes and closes. You cannot add to it afterwards.
            </p>
            <div className="dialog-buttons">
              <Dialog.Close asChild><button className="btn" autoFocus>Keep going</button></Dialog.Close>
              <button className="btn btn-primary" onClick={() => { studio.send({ cmd: "eval_end" }); setConfirm(false); }}>End eval</button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </section>
  );
}

function Judge({ rec, toJudge, judged, running }: { rec: EvalRecord; toJudge: boolean; judged: boolean; running: boolean }) {
  const run = useStudio((s) => s.telemetry?.policy ?? null);
  const control = useStudio((s) => s.control);
  const [note, setNote] = useState("");
  // A double press sends once. Only within a second, so a judgement the server refused can be retried.
  const sent = useRef({ id: "", at: 0 });
  const next = rec.n + 1;
  const mark = (outcome: "success" | "failure") => {
    if (!run || (sent.current.id === run.run_id && performance.now() - sent.current.at < 1000)) return;
    sent.current = { id: run.run_id, at: performance.now() };
    // The server judges its own copy of the run; run_id says which run this window showed.
    studio.send({ cmd: "eval_mark", outcome, note: note.trim(), run_id: run.run_id });
    setNote("");
  };
  // S and F judge, unless the operator is typing a note.
  useEffect(() => {
    if (!toJudge || !control) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.metaKey || e.ctrlKey) return;
      // WHY: an open dialog (End the eval?) holds focus; a habitual F there must not judge the episode behind it.
      if (e.target instanceof Element && e.target.closest('[role="dialog"], [role="alertdialog"]')) return;
      if (e.key === "s" || e.key === "S") mark("success");
      if (e.key === "f" || e.key === "F") mark("failure");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }); // eslint-disable-line react-hooks/exhaustive-deps

  const last = rec.episodes[rec.episodes.length - 1];
  return (
    <section className={`panel judge ${toJudge ? "is-open" : ""}`}>
      <div className="panel-head">
        <div>
          <h2 className="panel-title">{toJudge ? `Did episode ${next} succeed?` : running ? `Episode ${next} is running` : `Episode ${next}`}</h2>
          <p className="panel-sub">
            {toJudge && run ? `Ended after ${run.episode_s.toFixed(1)} s: ${run.ended ?? "stopped"}. Judge what the arm did, then reset the scene.`
              : running ? "Watch it. Judge it when it ends."
              : judged && last && rec.n >= rec.planned ? `Episode ${last.n} judged ${last.outcome}. That was the last planned episode. End the eval, or reset the scene to run more.`
              : judged && last ? `Episode ${last.n} judged ${last.outcome}. Reset the scene before episode ${next}.`
              : `Studio asks for the judgement when episode ${next} ends.`}
          </p>
        </div>
        {rec.n > 0 && (
          <button className="btn btn-ghost btn-sm" onClick={() => studio.send({ cmd: "eval_undo" })} disabled={!control}
            title={`Remove the judgement of episode ${rec.n}`}>
            <Undo2 aria-hidden /> Undo last
          </button>
        )}
      </div>
      {toJudge && (
        <div className="panel-body judge-body">
          <div className="judge-buttons">
            <button className="btn judge-btn tone-ok" onClick={() => mark("success")} disabled={!control}>
              <Check aria-hidden /> Success <span className="kbd">S</span>
            </button>
            <button className="btn judge-btn tone-danger" onClick={() => mark("failure")} disabled={!control}>
              <X aria-hidden /> Failure <span className="kbd">F</span>
            </button>
          </div>
          <input className="input" placeholder="Note, optional: what happened" value={note} maxLength={2000}
            onChange={(e) => setNote(e.target.value)} />
        </div>
      )}
      {running && run && (
        <div className="panel-body">
          <div className="progress" aria-hidden><span style={{ width: `${Math.min(100, (run.episode_s / run.limit_s) * 100)}%` }} /></div>
          <div className="progress-label num"><span>{run.episode_s.toFixed(1)} s</span><span className="faint">of {run.limit_s.toFixed(0)} s</span></div>
        </div>
      )}
    </section>
  );
}

function Episodes({ rec }: { rec: EvalRecord }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Episodes</h2>
        <span className="panel-sub num">{rec.successes} succeeded, {rec.n - rec.successes} failed</span>
      </div>
      {rec.episodes.length === 0 ? <p className="empty">None judged yet.</p> : (
        <ol className="episodes">
          {[...rec.episodes].reverse().map((e) => (
            <li key={e.n} className="episode">
              <span className="episode-n num">{e.n}</span>
              <span className={`badge tone-${e.outcome === "success" ? "ok" : "danger"}`}>
                {e.outcome === "success" ? "Success" : "Failure"}
              </span>
              <span className="episode-time num faint">{e.duration_s === null ? "" : `${e.duration_s.toFixed(1)} s`}</span>
              {e.note && <span className="episode-note">{e.note}</span>}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

// The point estimate with its interval drawn on a 0 to 100% track, so a wide interval looks wide.
function Interval({ k, n, lo, hi, compact }: { k: number; n: number; lo: number; hi: number; compact?: boolean }) {
  if (n === 0) return <span className="faint">{compact ? "None judged" : "No episodes yet"}</span>;
  const rate = k / n;
  return (
    <div className={`interval ${compact ? "is-compact" : ""}`}>
      <span className={compact ? "num strong" : "t-stat num"}>{pct(rate)}</span>
      <div className="interval-track" aria-hidden>
        <span className="interval-band" style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }} />
        <span className="interval-dot" style={{ left: `${rate * 100}%` }} />
      </div>
      <span className="num muted">{pct(lo)} to {pct(hi)}</span>
    </div>
  );
}
