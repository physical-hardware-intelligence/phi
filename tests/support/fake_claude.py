"""A stand-in for the `claude` CLI in its stream-json mode, for tests and for trying the assistant
panel without a Claude login. Reads one JSON user message per line; answers by keyword:

    crash    exit 3 with a message on stderr
    signout  answer like a signed-out CLI does (checked 2026-10-04)
    silent   never answer
    slow     stream slowly (for Stop)
    demo     a markdown answer with a list, code and a file path, streamed (for the browser check)
    anything else: echo the question, report two context fields, and "read" a file
"""

import json
import sys
import time


def out(ev: dict) -> None:
    sys.stdout.write(json.dumps(ev) + "\n")
    sys.stdout.flush()


def say(text: str, delay: float = 0.0) -> None:
    out({"type": "stream_event", "event": {"type": "message_start"}})
    for word in text.split(" "):
        out({"type": "stream_event", "event": {"type": "content_block_delta",
             "delta": {"type": "text_delta", "text": word + " "}}})  # fmt: skip
        time.sleep(delay)
    out({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}})


DEMO = (
    "**Likely cause.** The follower answered on a different port than "
    "`robot-config.yaml` lists.\n\n"
    "- Studio read `phi_follower` on `/dev/tty.usbmodem5B7B0096441`.\n"
    "- The identity check is in src/phi/studio/worker.py:303.\n\n"
    "**Fix.** Re-plug the follower, then read the arms again:\n\n"
    "```\nlerobot-find-port\n```"
)


def main() -> None:
    if "auth" in sys.argv and "status" in sys.argv:
        print(json.dumps({"loggedIn": "--signed-out" not in sys.argv, "authMethod": "stand-in"}))
        return
    turns = 0
    for line in sys.stdin:
        msg = json.loads(line)["message"]["content"]
        question = msg.rsplit("</studio_context>", 1)[-1].strip()
        turns += 1
        out({"type": "system", "subtype": "init", "tools": ["Read", "Grep", "Glob"]})
        if question == "crash":
            sys.stderr.write("stand-in: simulated crash\n")
            sys.exit(3)
        if question == "silent":
            time.sleep(3600)
        if question == "signout":
            text = "Failed to authenticate: OAuth session expired and could not be refreshed"
            out({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}})
            out({"type": "result", "subtype": "success", "is_error": False, "result": text})
            continue
        out({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read",
             "input": {"file_path": "robot-config.yaml"}}]}})  # fmt: skip
        if "demo" in question.lower():
            say(DEMO, delay=0.04)
            out({"type": "result", "subtype": "success", "is_error": False, "result": "ok",
                 "duration_ms": 2400})  # fmt: skip
            continue
        state = "state" in msg and '"state"' in msg
        replayed = "<earlier_turns>" in msg
        say(f"Turn {turns}. You asked: {question}. Context has state: {state}. "
            f"Replayed: {replayed}. See src/phi/studio/worker.py:303.",
            delay=0.3 if question == "slow" else 0.0)  # fmt: skip
        out({"type": "result", "subtype": "success", "is_error": False, "result": "ok",
             "total_cost_usd": 0.0, "duration_ms": 5})  # fmt: skip


if __name__ == "__main__":
    main()
