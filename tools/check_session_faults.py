#!/usr/bin/env python3
"""C02: real GPU sessions, opt-in fault server, exact ownership/recovery checks."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


def request(base, route, payload=None, headers=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + route, data=data, headers={
        "Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            try:
                body = response.read()
            except http.client.IncompleteRead as exc:
                return response.status, exc.partial.decode(errors="replace"), "incomplete"
            return response.status, body.decode(), None
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(), None


def metrics(base):
    status, body, error = request(base, "/metrics")
    assert status == 200 and error is None
    return {parts[0]: float(parts[1]) for line in body.splitlines()
            if (parts := line.split()) and len(parts) == 2 and not line.startswith("#")}


def idle(base):
    deadline = time.monotonic() + 15
    while True:
        result = metrics(base)
        keys = ["gem16_active_requests", "gem16_request_queue_active",
                "gem16_request_queue_depth", "gem16_pending_session_creations",
                "gem16_active_response_ids"]
        if all(result[key] == 0 for key in keys):
            return result
        if time.monotonic() >= deadline:
            raise AssertionError({key: result[key] for key in keys})
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Existing matrix result supplies immutable model paths and startup flags")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18085)
    parser.add_argument("--production", action="store_true",
                        help="Verify production ignores fault headers and still serves all request modes")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    base = f"http://127.0.0.1:{args.port}"
    report = {"scope": "C02 production hook exclusion" if args.production else "C02 injected lifecycle ownership and GPU recovery",
              "server_sha256": hashlib.sha256(args.server.read_bytes()).hexdigest(),
              "profiles": []}
    for config in json.loads(args.baseline.read_text())["profiles"]:
        command = config["server_command"].copy()
        command[0] = str(args.server.resolve())
        command[command.index("--port") + 1] = str(args.port)
        row = {"profile": config["profile"], "command": command, "cases": []}
        report["profiles"].append(row)
        log_path = args.output_dir / (config["profile"] + "-server.txt")
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("fault server exited during startup")
                    try:
                        initial = idle(base)
                        assert ("gem16_test_faults_observed" in initial) != args.production, "wrong executable mode"
                        break
                    except (OSError, urllib.error.URLError):
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.25)
                for responses in (False, True):
                    route = "/v1/responses" if responses else "/v1/chat/completions"
                    for stream in (False, True):
                        for resident in (False, True):
                            stages = ["acquired", "generation", "generation_after", "serialization"]
                            if not resident:
                                stages += ["reserved", "before_publication", "after_publication"]
                            if responses:
                                stages += ["index_before", "index_after", "commit_begin", "commit_messages", "commit_done"]
                            faults = [f"{stage}:{mode}" for stage in stages
                                      for mode in ("exception", "bad_alloc")]
                            faults += ["generation:unknown"]
                            if not resident:
                                faults += ["generation_status"]
                            if args.production:
                                faults = ["acquired:exception"]
                            for fault in faults:
                                case = {"route": route, "stream": stream,
                                        "resident": resident, "fault": fault}
                                row["cases"].append(case)
                                session_id = f"fault-{len(row['cases'])}"
                                headers = {"X-Gem16-Session-Id": session_id}
                                payload = {"model": "gem16", "reasoning_effort": "none"}
                                if responses:
                                    payload.pop("reasoning_effort")
                                    payload["reasoning"] = {"effort": "none"}
                                    payload.update(input="Reply only OK.", max_output_tokens=8)
                                else:
                                    payload.update(messages=[{"role": "user", "content": "Reply only OK."}],
                                                   max_completion_tokens=8)
                                if resident:
                                    code, body, error = request(base, route, payload, headers)
                                    assert code == 200 and error is None, body
                                    warm = json.loads(body)
                                    if responses:
                                        payload["previous_response_id"] = warm["id"]
                                    else:
                                        payload["messages"] += [warm["choices"][0]["message"],
                                                                 {"role": "user", "content": "Reply OK again."}]
                                before = idle(base)
                                payload["stream"] = stream
                                code, body, error = request(base, route, payload,
                                                            {**headers, "X-Gem16-Test-Fault": fault})
                                after = idle(base)
                                case.update(http_status=code, transport_error=error,
                                            body=body, before=before, after=after)
                                if args.production:
                                    assert code == 200 and error is None, case
                                    if stream:
                                        assert ("response.completed" if responses else "[DONE]") in body, case
                                else:
                                    assert after["gem16_test_faults_observed"] == before["gem16_test_faults_observed"] + 1, case
                                released = after["gem16_session_releases_total"] - before["gem16_session_releases_total"]
                                expected = 0 if fault.split(":")[0] in ("reserved", "before_publication") else 1
                                assert released == expected, (fault, released, expected)
                                if not args.production:
                                    assert after["gem16_indexed_responses"] == 0, case
                                if not args.production and fault != "generation_status":
                                    assert after["gem16_resident_sessions"] == 0, case
                                # Reuse the failed chat identity; rebuild only when discarded.
                                recovery = {"model": "gem16", "reasoning_effort": "none",
                                            "messages": [{"role": "user", "content": "Reply only OK."}],
                                            "max_completion_tokens": 8}
                                code, body, error = request(base, "/v1/chat/completions", recovery, headers)
                                assert code == 200 and error is None, body
                                answer = json.loads(body)
                                assert answer["usage"]["completion_tokens"] > 0, answer
                                recovered = idle(base)
                                assert recovered["gem16_session_releases_total"] == after["gem16_session_releases_total"] + 1
                                case["recovery"] = answer
                                case["status"] = "passed"
                                if len(row["cases"]) % 10 == 0:
                                    print(config["profile"], len(row["cases"]), "passed", flush=True)
                row["status"] = "passed"
            except Exception as exc:
                row.update(status="failed", error=repr(exc))
                print(config["profile"], "FAILED", repr(exc)[:1000], flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(row.get("status") == "passed" for row in report["profiles"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
