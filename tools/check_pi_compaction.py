import json, os, pathlib, queue, subprocess, tempfile, threading, time, sys
import argparse
import shutil
import urllib.request

parser = argparse.ArgumentParser(
    description="Run real Pi RPC compaction and new-session regression"
)
parser.add_argument("--base-url", required=True)
parser.add_argument("--models-file", type=pathlib.Path, required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--extended", action="store_true")
parser.add_argument("--resume-session", type=pathlib.Path, help="Cold-resume a COPY of a retained fixture session")
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
                    "reserveTokens": 12288 if args.extended else 1024,
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
    if args.extended or args.resume_session:
        cmd.remove("--no-session")
    restored_cwd = None
    if args.resume_session:
        header = json.loads(args.resume_session.read_text(encoding="utf-8").splitlines()[0])
        original_cwd = pathlib.Path(header["cwd"]).resolve()
        assert original_cwd.parent == pathlib.Path(tempfile.gettempdir()).resolve() and original_cwd.name.startswith("gem16-pi-compact-"), "not a disposable harness session"
        if not original_cwd.exists():
            original_cwd.mkdir()
            restored_cwd = original_cwd
        shutil.copy2(args.resume_session, p / "resume.jsonl")
        cmd += ["--session", str(p / "resume.jsonl")]
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
            text=True, encoding="utf-8",
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
                try:
                    event = events.get(timeout=min(.2, max(.01, deadline - time.monotonic())))
                except queue.Empty:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Pi exited {proc.returncode}; see stderr.txt")
                    continue
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

        report = {"command": cmd, "status": "failed", "client_config": config, "compaction_reserve_tokens": 12288 if args.extended else 1024}
        def rpc(kind, **fields):
            return send({"id": str(time.monotonic_ns()), "type": kind, **fields}, "response")["data"]
        def prompt(text):
            return send({"id": str(time.monotonic_ns()), "type": "prompt", "message": text}, "agent_end")

        try:
            if args.resume_session:
                expected = json.loads(args.resume_session.read_text(encoding="utf-8").splitlines()[0])["id"]
                report["cold_resume_source"] = str(args.resume_session)
                report["cold_resume_state"] = rpc("get_state")
                assert report["cold_resume_state"]["sessionId"] == expected
                report["cold_resume_answer"] = prompt("What was our codeword? Reply with that word only.")
                assert "ORCHID" in json.dumps(report["cold_resume_answer"])
                report["status"] = "passed"
            else:
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
                if args.extended:
                    parent = rpc("get_state")
                    entries = rpc("get_fork_messages")["messages"]
                    report["parent_state"] = parent
                    report["fork_result"] = rpc("fork", entryId=entries[-1]["entryId"])
                    fork = rpc("get_state")
                    assert fork["sessionId"] != parent["sessionId"]
                    report["fork_state"] = fork
                    report["fork_answer"] = prompt("For this branch only, replace the codeword with LILAC. What is the codeword now? Reply briefly.")
                    assert "LILAC" in json.dumps(report["fork_answer"])
                    report["resume_result"] = rpc("switch_session", sessionPath=parent["sessionFile"])
                    resumed = rpc("get_state")
                    assert resumed["sessionId"] == parent["sessionId"]
                    report["resume_answer"] = prompt("What was our codeword? Reply with that word only.")
                    assert "ORCHID" in json.dumps(report["resume_answer"])
                    assert "LILAC" not in json.dumps(report["resume_answer"])
                    report["resume_state"] = resumed
                send({"id": "new", "type": "new_session"}, "response")
                if args.extended:
                    fresh = rpc("get_state")
                    assert fresh["sessionId"] not in (parent["sessionId"], fork["sessionId"])
                    assert fresh["messageCount"] == 0
                    report["new_state"] = fresh

                report["new"] = send(
                    {"id": "newprompt", "type": "prompt", "message": "Say hello."},
                    "agent_end",
                )
                assert "ORCHID" in json.dumps(report["after"]), "compaction lost codeword"
                if args.extended:
                    send({"id": "enable-auto", "type": "set_auto_compaction", "enabled": True}, "response")
                    begin_events = len(transcript)
                    report["auto_seed"] = prompt("Remember the codeword ORCHID. " + ("The project uses Python and has two passing tests. " * 500) + " Reply briefly.")
                    # Pi emits agent_end before its background threshold compaction ends.
                    until = time.monotonic() + 120
                    while True:
                        ended = [json.loads(line) for line in transcript[begin_events:] if '"compaction_end"' in line]
                        if any(e.get("reason") == "threshold" for e in ended):
                            state = rpc("get_state")
                            if not state["isStreaming"] and not state["isCompacting"]:
                                break
                        if time.monotonic() >= until:
                            raise TimeoutError("automatic compaction completion not observed")
                        time.sleep(.02)
                    report["auto_after"] = prompt("What was the codeword and project language? Answer briefly.")
                    automatic = [json.loads(line) for line in transcript[begin_events:] if '"compaction_' in line]
                    report["automatic_events"] = automatic
                    assert any(e.get("type") == "compaction_start" and e.get("reason") == "threshold" for e in automatic), "automatic threshold compaction not observed"
                    assert any(e.get("type") == "compaction_end" and e.get("reason") == "threshold" and not e.get("aborted") and e.get("result") for e in automatic), "automatic compaction did not complete"
                    assert "ORCHID" in json.dumps(report["auto_after"])
                report["status"] = "passed"
        except Exception as e:
            report["error"] = repr(e)
        finally:
            proc.terminate()
            proc.wait(timeout=10)
            if restored_cwd is not None:
                restored_cwd.rmdir()
            (out / "transcript.jsonl").write_text("".join(transcript), encoding="utf-8")
            if args.extended and (p / "sessions").exists():
                shutil.copytree(p / "sessions", out / "sessions")
            (out / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"status": report["status"], "error": report.get("error")}))

sys.exit(0 if report["status"] == "passed" else 1)
