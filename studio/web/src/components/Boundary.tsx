import { CircleAlert, RotateCw } from "lucide-react";
import { Component, type ReactNode } from "react";
import { studio } from "../lib/studio";

interface Props { what: string; className?: string; children: ReactNode }

/** Keeps a crash in one part of the window from blanking the rest, the top bar and Stop included.
 * WHY: React unmounts the whole tree on an uncaught render error, and the commonest one here is a lazy
 * chunk that a rebuild deleted while this window was open. */
export class Boundary extends Component<Props, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) { return { error }; }

  componentDidCatch(error: Error): void {
    studio.localError(`${this.props.what} could not be shown: ${error.message}`, rebuilt(error)
      ? "Studio's interface was rebuilt since this window opened. Reload the window."
      : "Reload the window. If it repeats, ask Claude about this error.");
  }

  render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className={`boundary ${this.props.className ?? ""}`} role="alert">
        <div className="boundary-title"><CircleAlert aria-hidden /> {this.props.what} could not be shown</div>
        <p>
          {rebuilt(error)
            ? "Studio's interface was rebuilt since this window opened, so part of the old one is gone."
            : error.message}
        </p>
        <p className="faint">Stop and Esc still work. If this window has control, reloading stops the rig.</p>
        <div className="boundary-actions">
          <button className="btn btn-sm" onClick={() => location.reload()}><RotateCw aria-hidden /> Reload window</button>
          {/* WHY not after a rebuild: React.lazy keeps the failed import, so only a reload helps. */}
          {!rebuilt(error) && (
            <button className="btn btn-sm btn-ghost" onClick={() => this.setState({ error: null })}>Try again</button>
          )}
        </div>
      </div>
    );
  }
}

const rebuilt = (e: Error) => /dynamically imported module|Failed to fetch|Importing a module script failed|chunk/i.test(e.message);
