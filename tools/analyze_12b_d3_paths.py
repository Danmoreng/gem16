#!/usr/bin/env python3
"""Read-only D3 diagnostic accounting; raw capture rates are not primary TPS."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import sqlite3


def union_us(events):
    intervals = sorted((e["start"], e["end"]) for e in events)
    total = 0.0
    end = float("-inf")
    for start, stop in intervals:
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def native_capture(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    names = {r["id"]: r["value"] for r in db.execute("select * from StringIds")}
    kernels = []
    for row in db.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL order by start"):
        kernels.append({"start": row["start"] / 1000,
            "end": row["end"] / 1000, "name": names[row["demangledName"]],
            "short": names[row["shortName"]], "mangled": names[row["mangledName"]],
            "grid": [row["gridX"], row["gridY"], row["gridZ"]],
            "block": [row["blockX"], row["blockY"], row["blockZ"]],
            "registers": row["registersPerThread"],
            "shared": row["staticSharedMemory"] + row["dynamicSharedMemory"],
            "local_bytes_per_thread": row["localMemoryPerThread"],
            "stream": row["streamId"], "correlation": row["correlationId"],
            "graph": row["graphId"]})
    apis = [{"start": row["start"] / 1000, "end": row["end"] / 1000,
             "name": names[row["nameId"]], "correlation": row["correlationId"]}
            for row in db.execute("select * from CUPTI_ACTIVITY_KIND_RUNTIME")]
    db.close()
    return {"kernels": kernels, "apis": apis}


def inventory(events):
    signatures = collections.defaultdict(lambda: [0, 0.0])
    for event in events:
        value = signatures[event["name"]]
        value[0] += 1
        value[1] += event["end"] - event["start"]
    return [{"name": name, "count": count, "sum_us": duration}
            for name, (count, duration) in sorted(signatures.items(),
                                                key=lambda p: -p[1][1])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-sqlite", type=Path, required=True)
    parser.add_argument("--native-result", type=Path, required=True)
    parser.add_argument("--vllm-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = native_capture(args.native_sqlite)
    raw = args.output.with_name("native-kernel-events.json.gz")
    raw.write_bytes(gzip.compress(json.dumps(native).encode(), mtime=0))
    trace = json.loads(gzip.decompress(args.vllm_trace.read_bytes()))["traceEvents"]
    annotations = [e for e in trace if e.get("cat") == "user_annotation"]
    vllm = [{"name": e["name"], "start": e["ts"], "end": e["ts"] + e["dur"]}
            for e in trace if e.get("cat") == "kernel"]
    result = {"status": "diagnostic_capture_inventory_not_primary_throughput",
        "native_raw_events": str(raw),
        "native_kernel_counts": dict(collections.Counter(e["short"] for e in native["kernels"])),
        "native_dispatches": inventory(native["kernels"]),
        "vllm_dispatches": inventory(vllm),
        "vllm_annotation_counts": dict(collections.Counter(e["name"] for e in annotations)),
        "native_profile_output_count": len(json.loads(args.native_result.read_text())["representative_output_token_ids"])}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved exact native GPU/API events and bounded vLLM dispatch inventory")


if __name__ == "__main__":
    main()
