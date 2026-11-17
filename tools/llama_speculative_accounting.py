"""Offline per-request deductions from source-defined cumulative server counters.

No event histogram is invented from HTTP totals. Subtraction of cumulative
statistics is valid only for one sequential slot and one known implementation.
"""
import re

COUNTERS = re.compile(
    r"#calls\(b,g,a\) =\s+(\d+)\s+(\d+)\s+(\d+), #gen drafts =\s+(\d+), "
    r"#acc drafts =\s+(\d+), #gen tokens =\s+(\d+), #acc tokens =\s+(\d+)"
)


def request_deltas(log):
    previous = (0,) * 7
    result = []
    for match in COUNTERS.finditer(log):
        current = tuple(map(int, match.groups()))
        delta = tuple(a - b for a, b in zip(current, previous))
        begins, generates, accepts, drafted_groups, accepted_groups, proposed, accepted = delta
        if (begins != 1 or min(delta) < 0 or generates != accepts
                or accepted_groups > drafted_groups or drafted_groups > generates
                or accepted > proposed):
            raise ValueError("nonsequential or inconsistent cumulative speculative statistics")
        # At maximum two, accepted_groups counts calls with at least one
        # accepted token (verified against common_speculative_accept source).
        histogram = [drafted_groups - accepted_groups,
                     2 * accepted_groups - accepted, accepted - accepted_groups]
        if min(histogram) < 0:
            raise ValueError("acceptance counters incompatible with maximum depth two")
        depth = None
        if generates == drafted_groups and drafted_groups <= proposed <= 2 * drafted_groups:
            depth = {"1": 2 * drafted_groups - proposed,
                     "2": proposed - drafted_groups}
        result.append({
            "observed_cumulative": list(current), "observed_request_delta": list(delta),
            "derived": {"generated_groups": drafted_groups,
                        "acceptance_histogram_0_1_2": histogram,
                        "depth_histogram": depth},
            "source": "sequential-slot server aggregate counter differences; not an event trace",
        })
        previous = current
    return result
