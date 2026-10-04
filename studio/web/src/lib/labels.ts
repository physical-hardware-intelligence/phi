/** How a Studio key, joint or camera name reads on screen: left_follower becomes Left Follower,
 * shoulder_pan becomes Shoulder Pan. Keys stay snake_case everywhere else (commands, the server,
 * LeRobot), so only text a person reads goes through this. File names, ports, LeRobot ids and
 * commands are literal strings a person types or finds on disk, so they are never relabelled. */
export function label(key: string): string {
  return key
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

/** A list of names for a sentence: "Left Follower", "Left Follower and Right Follower". */
export function labels(keys: string[]): string {
  const l = keys.map(label);
  return l.length <= 2 ? l.join(" and ") : `${l.slice(0, -1).join(", ")} and ${l[l.length - 1]}`;
}
