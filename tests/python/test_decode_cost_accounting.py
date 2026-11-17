import importlib.util
from pathlib import Path
import unittest
import sqlite3
import tempfile


def module(name):
    path = Path(__file__).resolve().parents[2] / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


cost = module("analyze_decode_cost")
moe = module("decode_moe_accounting")


class DecodeCostAccountingTest(unittest.TestCase):
    def test_sqlite_attribution_preserves_global_merge_and_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.sqlite"
            with sqlite3.connect(path) as db:
                db.executescript("""
                    CREATE TABLE StringIds(id INTEGER,value TEXT);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,
                        shortName INTEGER,demangledName INTEGER,graphNodeId INTEGER,
                        streamId INTEGER,gridX INTEGER);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER);
                    CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER);
                    CREATE TABLE CUDA_GRAPH_NODE_EVENTS(graphNodeId INTEGER,originalGraphNodeId INTEGER);
                """)
                db.execute("INSERT INTO StringIds VALUES(0,'cudaGraphLaunch')")
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(0,1,0)")
                db.execute("INSERT INTO NVTX_EVENTS VALUES(0,100,'gem16.mtp.selection_sampling',NULL)")
                names = ["TiledEmbeddingLookupControlledKernel",
                         "MergeOnlineDecodeAttentionKernel", "MergeOnlineDecodeAttentionKernel"]
                for i, name in enumerate(names, 1):
                    full = name if i == 1 else name + f"<(int){256 if i == 2 else 512}, (int)128>"
                    db.execute("INSERT INTO StringIds VALUES(?,?)", (2*i, name))
                    db.execute("INSERT INTO StringIds VALUES(?,?)", (2*i+1, full))
                    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?)",
                               (10*i, 10*i+5, 2*i, 2*i+1, i, 1, 16))
            result, rows = cost.analyze(path)
            self.assertEqual(result["kernel_categories"]["attention_global_and_merge"]["count"], 1)
            self.assertEqual(result["kernel_categories"]["attention_local_and_merge"]["count"], 1)
            self.assertEqual(result["kernel_sum_ms"], 15 / 1e6)
            self.assertEqual(result["gpu_busy_union_ms"], 15 / 1e6)
            self.assertEqual(result["ordinary_tokens"], 1)
            self.assertFalse(result["unclassified_kernel_sum_ms"])

    def test_generic_merges_use_full_instance(self):
        name = "MergeOnlineDecodeAttentionKernel"
        for dim, kind in ((256, "local"), (512, "global")):
            full = f"void gem16::{name}<(int){dim}, (int)128>(...)"
            self.assertEqual(cost.classify(name, "", 1, "ordinary_attention", full),
                             f"attention_{kind}_and_merge")
        self.assertEqual(cost.classify(name, "", 1, "ordinary_attention"), "attention_unknown")
        self.assertEqual(cost.attention_kind(name, name + "<(int)128, (int)128>"), None)

    def test_generic_split_uses_full_26b_head_dimension(self):
        name = "SplitOnlineDecodeAttentionFp8Kernel"
        for dim, kind in ((256, "local"), (512, "global")):
            full = f"void gem16::{name}<(int){dim}, (int)8, (int)2, (int)2, (int)128, (bool)0, (bool)0>(...)"
            self.assertEqual(cost.classify(name, "", 16, "ordinary_attention", full),
                             f"attention_{kind}_and_merge")
        self.assertEqual(cost.classify(name, "", 16, "ordinary_attention"), "attention_unknown")

    def test_assistant_attention_is_separate(self):
        self.assertEqual(cost.classify("MergeOnlineDecodeAttentionKernel", "", 1, "assistant"),
                         "assistant_attention")

    def test_interval_union_preserves_overlap_and_idle(self):
        result = cost.sweep([(0, 8, "a"), (2, 4, "a"), (4, 10, "b")],
                            [(0, 12, "cudaStreamSynchronize")], 0, 12)
        self.assertEqual(result["gpu_busy_union_ms"], 10 / 1e6)
        self.assertEqual(result["gpu_exclusive_category_ms"], {"a": 4 / 1e6, "b": 2 / 1e6})
        self.assertEqual(result["gpu_overlap_ms"], 4 / 1e6)
        self.assertEqual(sum(result["gpu_idle_ms"].values()), 2 / 1e6)

    def test_partial_intervals_clip_at_window(self):
        result = cost.sweep([(-10, 5, "a"), (8, 20, "a")], [], 0, 10)
        self.assertEqual(result["gpu_busy_union_ms"], 7 / 1e6)
        self.assertEqual(sum(result["gpu_idle_ms"].values()), 3 / 1e6)

    def test_moe_terminal_branch_is_charged_once(self):
        def row(start, end, category):
            return dict(start=start, end=end, category=category, name=category, capture_scope="")
        rows = [row(0, 1, "input"), row(1, 5, "shared_mlp_down"),
                row(1, 3, "router"), row(3, 7, "trellis"), row(8, 9, "post")]
        result = moe.reconstruct(rows)
        self.assertEqual(result["winner"], "routed")
        self.assertEqual(result["span_ns"], 9)
        self.assertEqual(sum(result["critical_category_ns"].values()), 9)
        self.assertNotIn("shared_mlp_down", result["critical_category_ns"])

    def test_moe_rejects_join_before_inputs_complete(self):
        rows = [dict(start=a, end=b, category=c, name=c, capture_scope="")
                for a, b, c in [(0, 1, "input"), (1, 9, "shared_mlp_down"),
                                (1, 7, "trellis"), (8, 10, "post")]]
        with self.assertRaises(AssertionError):
            moe.reconstruct(rows)


if __name__ == "__main__":
    unittest.main()
