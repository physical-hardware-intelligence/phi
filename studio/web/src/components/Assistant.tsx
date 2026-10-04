import * as Dialog from "@radix-ui/react-dialog";
import {
  ArrowUp, Check, CircleAlert, Copy, FileText, FolderSearch, LoaderCircle, Search, Square, SquarePen, X,
} from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";
import { studio, useRoute, useStudio, type Part, type Turn } from "../lib/studio";
import { Markdown } from "./Markdown";

const PAGE_NAMES: Record<string, string> = {
  overview: "Overview", checks: "Checks", calibrate: "Calibrate", teleop: "Teleoperate", policy: "Run policy",
  evaluate: "Evaluate", files: "Files",
};

// Claude, beside every page. It reads what Studio sees and the files under Studio's folders; it has no
// way to send a rig command, so the panel never needs a confirmation step.
export function AssistantPanel() {
  const a = useStudio((s) => s.assist);
  const busy = a.turns.some((t) => t.state === "waiting" || t.state === "streaming");
  const body = useRef<HTMLDivElement>(null);
  const pinned = useRef(true); // follow new text unless the reader scrolled up

  useLayoutEffect(() => {
    const el = body.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [a.turns]);

  return (
    <aside className="assist" aria-label="Claude">
      <header className="assist-head">
        <div className="assist-heading">
          <h2 className="assist-title">Claude</h2>
          <Status />
        </div>
        <div className="assist-tools">
          <button className="btn btn-ghost btn-sm btn-icon" onClick={() => studio.newChat()} disabled={a.turns.length === 0}
            title="New conversation" aria-label="New conversation"><SquarePen aria-hidden /></button>
          <button className="btn btn-ghost btn-sm btn-icon" onClick={() => studio.closeAssistant()}
            title="Close (⌘J)" aria-label="Close Claude"><X aria-hidden /></button>
        </div>
      </header>
      <div className="assist-body" ref={body}
        onScroll={(e) => { const el = e.currentTarget; pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40; }}>
        <SignIn />
        {a.turns.length === 0 ? <Intro /> : a.turns.map((t, i) => <TurnView key={t.id} t={t} last={i === a.turns.length - 1} />)}
      </div>
      <Composer busy={busy} />
    </aside>
  );
}

function Status() {
  const st = useStudio((s) => s.assist.status);
  const [tone, text] = !st ? ["neutral", "Checking"] : st.available ? ["ok", st.model ? `Ready, ${st.model}` : "Ready"] : ["warn", "Not available"];
  return <span className={`assist-status tone-${tone}`}><span className="dot" />{text}</span>;
}

function SignIn() {
  const st = useStudio((s) => s.assist.status);
  if (!st || st.available) return null;
  const cmd = st.fix?.match(/claude auth login/) ? "claude auth login" : null;
  return (
    <div className="assist-signin">
      <div className="assist-signin-title"><CircleAlert aria-hidden /> {st.error}</div>
      {cmd ? (
        <>
          <p>Studio uses the Claude login on this Mac, so it stores no key. Sign in once in a terminal:</p>
          <CopyLine text={cmd} />
          <p className="faint">Then check again. Studio keeps working without Claude.</p>
        </>
      ) : st.fix && <p>{st.fix}</p>}
      <button className="btn btn-sm" onClick={() => studio.checkAssistant()}>Check again</button>
    </div>
  );
}

function CopyLine({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <div className="copy-line">
      <code>{text}</code>
      <button className="btn btn-ghost btn-sm btn-icon" aria-label="Copy command" title="Copy"
        onClick={() => { void navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); }}>
        {done ? <Check aria-hidden /> : <Copy aria-hidden />}
      </button>
    </div>
  );
}

function Intro() {
  const route = useRoute();
  const errors = useStudio((s) => s.errors);
  const fault = useStudio((s) => (s.state?.state === "FAULT" ? s.state.fault : null));
  const focus = useStudio((s) => s.assist.focus);
  const recent = fault ?? errors[errors.length - 1]?.message ?? null;
  const asks = [
    ...(focus ? ["Why did this happen, and how do I fix it?"] : recent ? [`Why did this happen, and how do I fix it: “${recent}”`] : []),
    "Which port is each arm on, and does that match robot-config.yaml?",
    "Does every arm match its own calibration file?",
    `What should I do next on the ${PAGE_NAMES[route] ?? route} page?`,
  ].slice(0, 4);
  return (
    <div className="assist-intro">
      <p>
        Ask about the rig, an error, or a file. Claude sees what Studio sees: the session, the arms and their
        ports, servo health, the recent log and this page. It can read Studio's folders. It cannot move the
        arms or change files.
      </p>
      <div className="assist-asks">
        {asks.map((q) => (
          <button key={q} className="assist-ask" onClick={() => studio.ask(q)}>{q}</button>
        ))}
      </div>
    </div>
  );
}

function TurnView({ t, last }: { t: Turn; last: boolean }) {
  const text = t.parts.filter((p) => p.kind === "text").map((p) => p.text).join("");
  return (
    <article className="turn">
      <div className="turn-q">
        {t.focus && <div className="turn-focus"><CircleAlert aria-hidden /><span>{t.focus.message}</span></div>}
        <p>{t.question}</p>
      </div>
      <div className="turn-a">
        {t.parts.map((p, i) => <PartView key={i} p={p} />)}
        {t.state === "waiting" && <div className="turn-wait"><LoaderCircle className="spin" aria-hidden /> Reading the rig state</div>}
        {t.state === "streaming" && <span className="caret" aria-hidden />}
        {t.state === "error" && t.error && (
          <div className="turn-error">
            <div className="turn-error-msg"><CircleAlert aria-hidden /> {t.error.message}</div>
            {t.error.fix && <div className="turn-error-fix">{t.error.fix}</div>}
            {last && <button className="btn btn-sm" onClick={() => studio.ask(t.question)}>Ask again</button>}
          </div>
        )}
        {(t.state === "done" || t.state === "stopped") && (
          <div className="turn-meta">
            <span>{t.state === "stopped" ? "Stopped" : t.ms ? `${(t.ms / 1000).toFixed(1)} s` : "Done"}</span>
            {text && <CopyButton text={text} />}
          </div>
        )}
      </div>
    </article>
  );
}

function PartView({ p }: { p: Part }) {
  if (p.kind === "text") return <Markdown text={p.text} />;
  const Icon = p.text.startsWith("Read ") ? FileText : p.text.startsWith("Searched") ? Search : FolderSearch;
  return <div className="turn-tool"><Icon aria-hidden /><span>{p.text}</span></div>;
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button className="btn btn-ghost btn-sm" onClick={() => { void navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); }}>
      {done ? <Check aria-hidden /> : <Copy aria-hidden />} {done ? "Copied" : "Copy"}
    </button>
  );
}

function Composer({ busy }: { busy: boolean }) {
  const focus = useStudio((s) => s.assist.focus);
  const open = useStudio((s) => s.assist.open);
  const route = useRoute();
  const [text, setText] = useState("");
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { if (open) box.current?.focus(); }, [open, focus]);
  useLayoutEffect(() => { // grow with the text, up to about eight lines
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [text]);

  const send = () => { if (studio.ask(text)) setText(""); };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); }
  };

  return (
    <div className="assist-compose">
      {focus && (
        <div className="compose-focus">
          <CircleAlert aria-hidden />
          <span className="ellipsis" title={focus.message}>About: {focus.message}</span>
          <button className="btn btn-ghost btn-sm btn-icon" onClick={() => studio.clearFocus()} aria-label="Remove"><X aria-hidden /></button>
        </div>
      )}
      <div className="compose-box">
        <textarea ref={box} className="compose-input" rows={1} value={text} maxLength={8000}
          placeholder={focus ? "What do you want to know about it?" : "Ask about the rig, an error, or a file"}
          onChange={(e) => setText(e.target.value)} onKeyDown={onKey} aria-label="Question for Claude" />
        {busy ? (
          <button className="btn btn-sm btn-icon compose-send" onClick={() => studio.stopAnswer()} aria-label="Stop the answer" title="Stop the answer">
            <Square aria-hidden />
          </button>
        ) : (
          <button className="btn btn-primary btn-sm btn-icon compose-send" onClick={send} disabled={!text.trim()} aria-label="Send" title="Send (Enter)">
            <ArrowUp aria-hidden />
          </button>
        )}
      </div>
      <div className="compose-foot">
        <span>Sees the rig state, recent log and the {PAGE_NAMES[route] ?? route} page.</span>
        <ContextDialog />
      </div>
    </div>
  );
}

function ContextDialog() {
  const ctx = useStudio((s) => s.assist.context);
  const [open, setOpen] = useState(false);
  return (
    <Dialog.Root open={open} onOpenChange={(o) => { setOpen(o); if (o) studio.showContext(); }}>
      <Dialog.Trigger asChild><button className="link-btn">Show</button></Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="dialog dialog-wide" aria-describedby="ctx-desc">
          <Dialog.Title className="dialog-title">What Claude receives with your next question</Dialog.Title>
          <p id="ctx-desc" className="dialog-text">Built by Studio when you send. The access token is never included.</p>
          <pre className="ctx-pre">{ctx ?? "Loading"}</pre>
          <div className="dialog-buttons"><Dialog.Close asChild><button className="btn">Close</button></Dialog.Close></div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
