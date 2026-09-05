import json, os, pathlib, queue, subprocess, tempfile, threading, time, sys
import argparse

parser = argparse.ArgumentParser(
    description="Run real Pi RPC compaction and new-session regression"
)
parser.add_argument("--base-url", required=True)
parser.add_argument("--models-file", type=pathlib.Path, required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
base, destination = args.base_url, args.output_dir
root = pathlib.Path(__file__).resolve().parents[1]
out = pathlib.Path(destination)
out.mkdir(exist_ok=False, parents=True)
config = json.loads(args.models_file.read_text())
config["providers"]["gem16"]["baseUrl"] = base + "/v1"
with tempfile.TemporaryDirectory(prefix="gem16-pi-compact-") as temp:
    p = pathlib.Path(temp)
    (p / "models.json").write_text(json.dumps(config))
    (p / "settings.json").write_text(
        json.dumps(
            {
                "compaction": {
                    "enabled": False,
                    "keepRecentTokens": 128,
                    "reserveTokens": 1024,
                }
            }
        )
    )
    cmd = [
        "node",
        str(
            root
            / "tools/pi-agent/node_modules/@earendil-works/pi-coding-agent/dist/cli.js"
        ),
        "--provider",
        "gem16",
        "--model",
        "gem16",
        "--thinking",
        "off",
        "--mode",
        "rpc",
        "--no-session",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
    ]
    events = queue.Queue()
    transcript = []
    with (out / "stderr.txt").open("w") as err:
        proc = subprocess.Popen(
            cmd,
            cwd=p,
            env={**os.environ, "PI_CODING_AGENT_DIR": str(p)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
        )

        def reader():
            for line in proc.stdout:
                transcript.append(line)
                try:
                    events.put(json.loads(line))
                except ValueError:
                    pass

        threading.Thread(target=reader, daemon=True).start()

        def send(message, terminal):
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                event = events.get(timeout=max(0.1, deadline - time.monotonic()))
                if event.get("type") == "response" and event.get("success") is False:
                    raise RuntimeError(event)
                if event.get("type") == terminal and (
                    terminal != "response" or event.get("id") == message["id"]
                ):
                    if terminal == "agent_end":
                        answers = [
                            m
                            for m in event.get("messages", [])
                            if m.get("role") == "assistant"
                        ]
                        assert (
                            answers and answers[-1].get("stopReason") == "stop"
                        ), event
                    return event
            raise TimeoutError(message)

        report = {"command": cmd, "status": "failed"}
        try:
            for i in range(3):
                send(
                    {
                        "id": str(i),
                        "type": "prompt",
                        "message": "Remember codeword ORCHID. "
                        + ("The project uses Python and has two passing tests. " * 100)
                        + " Reply briefly.",
                    },
                    "agent_end",
                )
            report["compaction"] = send(
                {
                    "id": "compact",
                    "type": "compact",
                    "customInstructions": "Retain the codeword and project facts.",
                },
                "response",
            )
            report["after"] = send(
                {
                    "id": "after",
                    "type": "prompt",
                    "message": "What was the codeword? Answer briefly.",
                },
                "agent_end",
            )
            send({"id": "new", "type": "new_session"}, "response")
            report["new"] = send(
                {"id": "newprompt", "type": "prompt", "message": "Say hello."},
                "agent_end",
            )
            assert "ORCHID" in json.dumps(report["after"]), "compaction lost codeword"
            report["status"] = "passed"
        except Exception as e:
            report["error"] = repr(e)
        finally:
            proc.terminate()
            proc.wait(timeout=10)
            (out / "transcript.jsonl").write_text("".join(transcript))
            (out / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))

sys.exit(0 if report["status"] == "passed" else 1)
