import { CameraOff } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { studio, useStudio } from "../lib/studio";
import { label } from "../lib/labels";

// Draws the newest frame only: if a decode is still running when the next frame arrives,
// the older one is dropped. Shows the measured preview rate and flags a stale feed.
export function CameraTile({ name }: { name: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [fps, setFps] = useState(0);
  const [stale, setStale] = useState(false);
  const [size, setSize] = useState<[number, number] | null>(null);
  const status = useStudio((s) => s.cameras[name]);
  const linked = useStudio((s) => s.link === "open");

  useEffect(() => {
    let busy = false;
    let last = 0;
    const arrivals: number[] = [];
    const off = studio.onFrames(name, async (f) => {
      const now = performance.now();
      arrivals.push(now);
      while (arrivals.length && now - arrivals[0] > 1000) arrivals.shift();
      last = now;
      if (busy || !canvas.current) return;
      busy = true;
      try {
        const bmp = await createImageBitmap(f.jpeg);
        const c = canvas.current;
        if (c) {
          const w = bmp.width, h = bmp.height; // read now: the updater below runs after close()
          if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
          setSize((s) => (s && s[0] === w && s[1] === h ? s : [w, h]));
          c.getContext("2d")!.drawImage(bmp, 0, 0);
        }
        bmp.close();
      } finally {
        busy = false;
      }
    });
    const timer = setInterval(() => {
      const now = performance.now();
      while (arrivals.length && now - arrivals[0] > 1000) arrivals.shift();
      setFps(arrivals.length);
      setStale(last > 0 && now - last > 500);
    }, 250);
    return () => { off(); clearInterval(timer); };
  }, [name]);

  const offline = status && !status.online;
  return (
    <figure className={`cam ${stale || offline ? "is-stale" : ""}`}>
      <canvas ref={canvas} width={640} height={480} aria-label={`${label(name)} camera`} />
      {(offline || !size) && (
        <div className="cam-empty">
          {offline ? <CameraOff aria-hidden /> : null}
          <span>{offline ? status?.message ?? "No signal" : linked ? "Waiting for frames" : "Not connected"}</span>
        </div>
      )}
      <figcaption className="cam-bar">
        <span className="cam-name">{label(name)}</span>
        <span className="num t-xs faint">
          {size ? `${size[0]}×${size[1]}` : ""}
        </span>
        <span className={`num t-xs ${stale ? "cam-warn" : "faint"}`}>
          {!linked ? "" : stale ? "stale" : `${fps} fps`}
        </span>
      </figcaption>
    </figure>
  );
}
