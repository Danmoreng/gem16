import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from analyze_12b_d3_paths import union_us
from compare_12b_d3_paths import family, native_groups, vllm_groups


class D3PathAccountingTests(unittest.TestCase):
    def test_offline_global_attention_and_merge_are_not_counted_as_norms(self):
        for name in ("gem16_d3_global_fp8", "void gem16::internal::MergeD3Fast16Kernel(float*)"):
            self.assertEqual(family({"name": name, "phase": "target backbone"}, "native"),
                             "global attention")
        for name in ("gem16_d3_local_fp8", "void gem16::internal::MergeD3Local4Kernel(float*)"):
            self.assertEqual(family({"name": name, "phase": "target backbone"}, "native"),
                             "local attention")

    def test_asynchronous_graph_nodes_use_launch_correlation_without_external_id(self):
        events = [
            {"cat": "user_annotation", "name": "execute_4_generation_1(sq4sk16414)", "ts": 0, "dur": 20},
            {"cat": "user_annotation", "name": "gem16_diag.assistant.propose", "ts": 40, "dur": 30},
            {"cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": 10, "args": {"correlation": 1}},
            {"cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": 50, "args": {"correlation": 2}},
        ]
        for correlation, when in [(1, 100), (2, 120)]:
            events.append({"cat": "kernel", "name": "graph node", "ts": when, "dur": 5,
                           "args": {"correlation": correlation, "grid": [1, 1, 1], "block": [128, 1, 1]}})
        group = vllm_groups(events)[0]
        self.assertEqual([k["phase"] for k in group["kernels"]], ["target backbone", "assistant"])

    def test_overlapping_kernel_intervals_are_not_added_as_device_busy_time(self):
        events = [{"start": 0, "end": 10}, {"start": 2, "end": 8}, {"start": 9, "end": 12}]
        self.assertEqual(union_us(events), 12)

    def test_derived_d3_acceptance_is_checked_against_same_capture_counters(self):
        kernels = []
        for repeat in range(4):
            for offset, name in enumerate(["InitializeControlledAssistantKernel", "BuildControlledMtpInputsKernel",
                                           "FusedOutputHeadFixedBatchCandidatesKernel", "AcceptMtpGroupKernel",
                                           "ContinueMtpChainKernel"]):
                kernels.append({"short": name, "name": name, "start": repeat * 10 + offset,
                                "end": repeat * 10 + offset + .5})
        result = {"runs": [{"engine_result": {"mtp": {"verification_groups": 2,
            "accepted_tokens": 4, "proposed_token_ids": [11, 999, 777, 33, 44, 55]}}}],
            "representative_output_token_ids": [0, 11, 22, 33, 44, 55, 66, 77]}
        groups = native_groups({"kernels": kernels}, result)
        self.assertEqual([g["accepted_derived"] for g in groups], [1, 3])
        result["runs"][0]["engine_result"]["mtp"]["accepted_tokens"] = 5
        with self.assertRaises(ValueError):
            native_groups({"kernels": kernels}, result)


if __name__ == "__main__":
    unittest.main()
