import { Check, Copy } from "lucide-react";
import { useState } from "react";

/** One flag per line with a trailing backslash, the way LeRobot's docs print them. The copied text is the
 * same, and a shell runs it as one command. */
export function CommandBlock({ cmd, wrap = true }: { cmd: string; wrap?: boolean }) {
  const text = wrap ? cmd.split(/ (?=--|'--)/).join(" \\\n  ") : cmd;
  const [done, setDone] = useState(false);
  return (
    <div className="cmd-block">
      <pre><code>{text}</code></pre>
      <button className="btn btn-ghost btn-sm btn-icon cmd-copy" aria-label="Copy command" title="Copy"
        onClick={() => { void navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); }}>
        {done ? <Check aria-hidden /> : <Copy aria-hidden />}
      </button>
    </div>
  );
}
