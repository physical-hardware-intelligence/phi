import { ArrowRight, TriangleAlert } from "lucide-react";
import { useEffect, type ReactNode } from "react";
import { CommandBlock } from "../components/CommandBlock";
import { go, useSection, type Route } from "../lib/studio";

// How to use Studio and set up an SO-101 rig. Every statement here is something the code does today (each
// page's file holds the behaviour); setup advice comes from docs/robots/so-arm101. The top bar's help button
// opens this page at the current page's section, as #/guide/<route>.

const TOC: { id: string; title: string }[] = [
  { id: "start", title: "Start here" },
  { id: "rig", title: "Set up a new SO-101 rig" },
  { id: "pages", title: "What each page does" },
  { id: "states", title: "Rig states" },
  { id: "safety", title: "Safety" },
  { id: "keys", title: "Keyboard" },
  { id: "trouble", title: "Troubleshooting" },
  { id: "where", title: "Where things are kept" },
];

export function Guide() {
  const section = useSection();
  // WHY scroll on the section, not on mount only: the help button can be pressed again from this page
  useEffect(() => {
    const el = section ? document.getElementById(`guide-${section}`) : null;
    if (el) el.scrollIntoView({ block: "start" });
    else document.querySelector(".content")?.scrollTo({ top: 0 });
  }, [section]);

  return (
    <div className="page guide">
      <nav className="guide-toc" aria-label="Guide contents">
        <div className="field-label">On this page</div>
        {TOC.map((t) => (
          <a key={t.id} href={`#/guide/${t.id}`} className={`guide-toc-item ${section === t.id ? "is-active" : ""}`}
            onClick={(e) => { e.preventDefault(); go("guide", t.id); }}>{t.title}</a>
        ))}
      </nav>
      <div className="guide-body">
        <Start />
        <Rig />
        <Pages />
        <States />
        <Safety />
        <Keys />
        <Trouble />
        <Where />
      </div>
    </div>
  );
}

function Section({ id, title, sub, children }: { id: string; title: string; sub?: string; children: ReactNode }) {
  return (
    <section className="panel guide-section" id={`guide-${id}`} aria-labelledby={`guide-${id}-title`}>
      <div className="panel-head">
        <h2 className="panel-title" id={`guide-${id}-title`}>{title}</h2>
        {sub && <span className="panel-sub">{sub}</span>}
      </div>
      <div className="panel-body guide-text">{children}</div>
    </section>
  );
}

function Open({ to, children }: { to: Route; children: ReactNode }) {
  return (
    <button className="btn btn-sm" onClick={() => go(to)}>
      {children} <ArrowRight aria-hidden />
    </button>
  );
}

function Steps({ items }: { items: { title: string; body: ReactNode; to?: Route; open?: string }[] }) {
  return (
    <ol className="guide-steps">
      {items.map((s, i) => (
        <li key={s.title}>
          <span className="guide-n num" aria-hidden>{i + 1}</span>
          <div className="guide-step">
            <div className="guide-step-title">{s.title}</div>
            <div className="guide-step-body">{s.body}</div>
          </div>
          {s.to && <Open to={s.to}>{s.open ?? "Open"}</Open>}
        </li>
      ))}
    </ol>
  );
}

const Kbd = ({ children }: { children: ReactNode }) => <span className="kbd">{children}</span>;
const Id = ({ children }: { children: ReactNode }) => <code className="ident">{children}</code>;

function Start() {
  return (
    <Section id="start" title="Start here" sub="From a fresh start to a judged policy run">
      <p className="guide-note">
        <TriangleAlert aria-hidden />
        <span>
          Studio runs a mock rig today: simulated arms and cameras, so every page works with no hardware. The real-arm
          backend is not built yet, and <Id>--hardware</Id> stops with a message. For real arms, use the LeRobot
          commands on the LeRobot setup page.
        </span>
      </p>
      <Steps items={[
        { title: "Start Studio", body: <>
            In the phi checkout run the command below. It prints a link and opens it in your browser. The link holds a
            token that changes on every launch, and Studio only listens on this Mac.
            <CommandBlock cmd="make studio" wrap={false} />
            Add options through <Id>STUDIO_ARGS</Id>, for example a bimanual mock rig on another port:
            <CommandBlock cmd='make studio STUDIO_ARGS="--pairs 2 --port 8766"' wrap={false} />
          </> },
        { title: "Take control", body: <>
            One window drives the rig at a time. Other windows are view only, and can still press Stop, ask Claude, read
            Files and run Checks. <strong>Take control</strong> in the top bar moves control to this window and stops
            any motion first.
          </> },
        { title: "Run the checks", to: "checks", body: <>
            Read-only checks of this Mac, the rig and Studio. Nothing moves and no servo register is written.
          </> },
        { title: "Connect and confirm the arms", to: "overview", body: <>
            <strong>Connect</strong> reads each arm's calibration registers and matches them to its calibration file.
            Nothing moves. If every arm matches, <strong>Confirm arms</strong>; if one does not, calibrate it.
          </> },
        { title: "Calibrate an arm that does not match", to: "calibrate", body: <>
            Four steps: check, set the middle, record ranges, review and save.
          </> },
        { title: "Teleoperate", to: "teleop", body: <>
            Enable torque, then <strong>Start teleop</strong>. Each follower follows its leader. A quick way to see the
            calibration is right.
          </> },
        { title: "Run a policy, then evaluate it", to: "evaluate", open: "Open Evaluate", body: <>
            Run policy drives the followers for a set time. Evaluate runs a set number of episodes and gives a success
            rate.
          </> },
      ]} />
    </Section>
  );
}

function Rig() {
  return (
    <Section id="rig" title="Set up a new SO-101 rig" sub="Once per rig, with the LeRobot commands">
      <Steps items={[
        { title: "Label the power adapters", body: <>
            The leader and follower ship with different adapters, one 5 V and one 12 V, and they look alike. The wrong
            one shows up as missing motors or <Id>Input voltage error!</Id>. Put the right adapter on, then power-cycle:
            a latched voltage error only clears on a power cycle.
          </> },
        { title: "Use a powered USB hub", body: <>
            Two arms and three cameras are more than a laptop's own USB bus should power; a bus-powered hub can brown
            out mid-recording.
          </> },
        { title: "Find each arm's port", to: "setup", open: "Commands", body: <>
            Run <Id>lerobot-find-port</Id> and unplug one arm when it asks: the port that disappears is that arm's. Put
            the ports in <Id>robot-config.yaml</Id>. A port name can change after a replug or reboot.
          </> },
        { title: "Set the motor ids", body: <>
            Once per new arm, with <Id>lerobot-setup-motors</Id>. Skip this for the assembled Kit Pro.
          </> },
        { title: "Calibrate each arm", to: "calibrate", body: <>
            At "set the middle", put every joint mid-range, and Wrist Roll at the centre of the range you will use with
            the gripper level. Wrist Roll has no stops: you get 180° either way from that pose and no more.
          </> },
        { title: "Fill in the camera indices", to: "checks", open: "Checks", body: <>
            Each camera in <Id>robot-config.yaml</Id> needs its <Id>index_or_path</Id>. Checks flags any still set to TBD.
          </> },
      ]} />
    </Section>
  );
}

const PAGE_HELP: { id: Route; title: string; what: string; how: ReactNode }[] = [
  { id: "overview", title: "Overview", what: "Rig status and the next step.", how: <>
      The workflow runs Connect, Confirm arms, Teleoperate, Run a policy, Evaluate; later steps unlock once the arms
      are confirmed. Rig health shows each arm's calibration, torque, temperature (warns at 60 °C) and load (warns
      above 80%). Activity lists this session's events, newest first.
    </> },
  { id: "checks", title: "Checks", what: "Every read-only check, in one run.", how: <>
      Runs on your first visit. <strong>Only problems</strong> hides what passed. A problem shows its fix when Studio
      knows one, and <strong>Ask Claude</strong> opens Claude with that check attached.
    </> },
  { id: "calibrate", title: "Calibrate", what: "Writes an arm's homing offsets and joint ranges.", how: <>
      <strong>Set middle</strong> makes the present pose read 2047. Then move every joint end to end except Wrist Roll;
      <strong> Finish recording</strong> unlocks once each has moved. Review old against new, then
      <strong> Save calibration</strong>. <strong>Cancel and restore</strong> writes the old registers back at any
      step. It will not start while an arm holds torque, a cable is swapped, or the rig has a fault.
    </> },
  { id: "teleop", title: "Teleoperate", what: "Each follower follows its leader.", how: <>
      <strong>Enable torque</strong>, then <strong>Start teleop</strong>: a 3 second countdown, which <Kbd>Esc</Kbd>{" "}
      cancels. A follower's goal is kept within 8° of where each joint is now (5 on the gripper's 0 to 100 scale); these
      limits are placeholders until tested on real arms. There is no time limit:
      stop with <Kbd>Esc</Kbd>.
    </> },
  { id: "policy", title: "Run policy", what: "A policy drives the followers for a set time.", how: <>
      Pick a policy, a task, and a time limit (1 to 600 s). <strong>Run policy</strong> counts down 3 seconds, and the
      run stops itself at the limit with the followers holding. To run again, <strong>Resume</strong>, then Run policy.
      Today only the mock's scripted policy runs; ACT and π0.5 checkpoints need the real-arm backend.
    </> },
  { id: "evaluate", title: "Evaluate", what: "Episodes, your judgement, a success rate.", how: <>
      Set the episode count and <strong>Start eval</strong>. Run each episode, then judge it with <Kbd>S</Kbd> for
      success or <Kbd>F</Kbd> for failure, with an optional note. <strong>Undo last</strong> takes a judgement back. The
      success rate comes with a 95% interval. Every judgement is saved at once, and an unfinished eval resumes the
      next time you start one.
    </> },
  { id: "setup", title: "LeRobot setup", what: "Your rig as LeRobot reads robot-config.yaml.", how: <>
      Each arm's type, port, calibration file, units and step limit; what LeRobot would refuse; the commands for this
      rig in order, with a copy button; and the dataset keys a recording writes.
    </> },
  { id: "files", title: "Files", what: "Read-only view of the files that matter when something breaks.", how: <>
      Serial ports with the arm on each, the rig notes, the calibration files, search, and a viewer. Files that may
      hold secrets are never shown.
    </> },
];

function Pages() {
  const section = useSection();
  return (
    <Section id="pages" title="What each page does">
      <div className="guide-pages">
        {PAGE_HELP.map((p) => (
          <div key={p.id} className={`guide-page ${section === p.id ? "is-target" : ""}`} id={`guide-${p.id}`}>
            <div className="guide-page-head">
              <h3 className="guide-page-title">{p.title}</h3>
              <Open to={p.id}>Open</Open>
            </div>
            <p className="guide-page-what">{p.what}</p>
            <p className="guide-page-how">{p.how}</p>
          </div>
        ))}
        <div className="guide-page" id="guide-claude">
          <div className="guide-page-head"><h3 className="guide-page-title">Ask Claude <Kbd>⌘J</Kbd></h3></div>
          <p className="guide-page-what">Questions about the rig, with Studio's live state attached.</p>
          <p className="guide-page-how">
            Claude can read files but cannot run commands, change files or move the arms. It needs Claude Code on this
            Mac, signed in. Chats are not saved.
          </p>
        </div>
      </div>
    </Section>
  );
}

const STATES: [string, string, string][] = [
  ["Offline", "No rig session", "Connect"],
  ["Connecting", "Reading each arm", "Wait"],
  ["Confirm arms", "Arms read, not yet confirmed", "Check each arm, then confirm"],
  ["Torque off", "Confirmed, followers limp", "Enable torque"],
  ["Holding", "Followers hold their pose", "Start teleop or a policy"],
  ["Moving", "Teleop or a policy is driving", "Esc to stop"],
  ["Stopped", "Held after a stop", "Resume, or turn torque off"],
  ["Fault", "A servo fault or a lost arm", "Read the fault, turn torque off, clear"],
  ["Calibrating", "A calibration is running", "Follow its steps"],
];

function States() {
  return (
    <Section id="states" title="Rig states" sub="The pill in the top bar">
      <div className="table-scroll">
        <table className="table guide-table">
          <thead><tr><th>State</th><th>Meaning</th><th>Next</th></tr></thead>
          <tbody>{STATES.map(([s, m, n]) => <tr key={s}><td className="strong">{s}</td><td>{m}</td><td>{n}</td></tr>)}</tbody>
        </table>
      </div>
    </Section>
  );
}

function Safety() {
  return (
    <Section id="safety" title="Safety">
      <ul className="guide-list">
        <li><strong>Stop</strong> (<Kbd>Esc</Kbd>, on every page and in every window) holds each follower where it is,
          with torque on. It is a software stop: the physical cut is the followers' power.</li>
        <li>Motion also stops when the controlling window stops answering for 1 second, closes or reloads, or hands
          control to another window.</li>
        <li><strong>Turning torque off</strong> lets the followers fall under their own weight. Support them, or lower
          them to rest first. Studio asks before it does this.</li>
        <li><strong>Ctrl+C</strong> in the terminal stops Studio and turns torque off on every arm, so support the
          followers first.</li>
        <li>Torque only comes on after the arms are confirmed, and Studio refuses an arm whose servos do not match its
          own calibration file.</li>
        <li>Keep teleop out of full stretch: near it, a small leader move asks for a large joint move and the arm jerks.</li>
      </ul>
    </Section>
  );
}

function Keys() {
  const rows: [ReactNode, string][] = [
    [<Kbd key="k">Esc</Kbd>, "Stop, on any page. Also cancels a countdown."],
    [<><Kbd>⌘J</Kbd> or <Kbd>Ctrl J</Kbd></>, "Open or close Claude."],
    [<><Kbd>S</Kbd> / <Kbd>F</Kbd></>, "Judge the waiting eval episode a success or a failure."],
    [<><Kbd>Enter</Kbd>, <Kbd>Shift Enter</Kbd></>, "Send to Claude; a new line."],
  ];
  return (
    <Section id="keys" title="Keyboard">
      <table className="table guide-table guide-keys">
        <tbody>{rows.map(([k, v], i) => <tr key={i}><td>{k}</td><td>{v}</td></tr>)}</tbody>
      </table>
    </Section>
  );
}

const TROUBLE: [string, ReactNode][] = [
  ["Sidebar says Studio is not running", <>Start it again with <Id>make studio</Id> and open the link it prints.</>],
  ["Sidebar says Not authorised", <>The token changes on every launch. Open the link Studio printed this time.</>],
  ["Port 8765 is in use", <>Another Studio is running. Close it, or pass another port: <Id>STUDIO_ARGS="--port 8766"</Id>.</>],
  ["View only: another window has control", <>Press <strong>Take control</strong> in the top bar.</>],
  ["An arm holds another arm's calibration", <>The USB cables are swapped. Swap them back, then <strong>Read again</strong>.</>],
  ["Missing motors, or Input voltage error!", <>Wrong power adapter. Swap the adapters, then power-cycle the arm.</>],
  ["The follower does not mirror the leader", <>Calibrate again with the same id, and check Each arm matches its calibration on Checks.</>],
  ["Wrist Roll snaps 90 to 180°", <>It crossed its seam, 180° from the calibration pose. Calibrate with the wrist level and stay within that range.</>],
  ["A policy reaches a centimetre off on another Mac", <>The calibrations differ. Compare them: <Id>python -m phi.utils.compare_calibration ID OTHER</Id>.</>],
  ["Motion stopped because this window stopped answering", <>This window missed its heartbeat for over a second. Check it is connected, then Resume.</>],
  ["Claude says it is not installed, not signed in, or too old", <>Install Claude Code, run <Id>claude auth login</Id>, or run <Id>claude update</Id>.</>],
];

function Trouble() {
  return (
    <Section id="trouble" title="Troubleshooting" sub="More in docs/robots/so-arm101/troubleshooting.md">
      <div className="table-scroll">
        <table className="table guide-table guide-trouble">
          <thead><tr><th>You see</th><th>What to do</th></tr></thead>
          <tbody>{TROUBLE.map(([s, f]) => <tr key={s}><td className="strong">{s}</td><td>{f}</td></tr>)}</tbody>
        </table>
      </div>
    </Section>
  );
}

function Where() {
  const rows: [ReactNode, string][] = [
    [<Id key="r">robot-config.yaml</Id>, "Ports, ids and cameras for this Mac, in the main checkout. Git ignores it."],
    [<Id key="c">~/.cache/huggingface/lerobot/calibration</Id>, "LeRobot's calibration files, one per arm id ($HF_LEROBOT_CALIBRATION moves it)."],
    [<Id key="d">~/.cache/phi/studio</Id>, "Studio's data: evals, and the mock rig's calibrations. --data-dir moves it."],
    ["Not saved", "Telemetry, the activity list and Claude chats live in memory only."],
  ];
  return (
    <Section id="where" title="Where things are kept">
      <table className="table guide-table">
        <tbody>{rows.map(([k, v], i) => <tr key={i}><td>{k}</td><td>{v}</td></tr>)}</tbody>
      </table>
    </Section>
  );
}
