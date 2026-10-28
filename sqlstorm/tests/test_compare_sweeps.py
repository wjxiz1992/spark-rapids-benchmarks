# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest

from sqlstorm.common import atomic_json, digest
from sqlstorm.compare_sweeps import compare, records
from sqlstorm.compare_sweeps import order_review


class Sweeps(unittest.TestCase):
    def test_progress_reader_ignores_an_in_flight_journal_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "results.jsonl").write_text('{"id":"1"}\n{"id":')
            self.assertEqual({"1": {"id": "1"}}, records(root))

    def test_ordered_output_difference_requires_review_instead_of_a_multiset_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            c, g, sql = [root / name for name in ("cpu", "gpu", "q.sql")]
            c.write_text('[1]\n[2]\n'); g.write_text('[2]\n[1]\n')
            sql.write_text('SELECT x FROM t ORDER BY x')
            self.assertEqual("requires_order_review", order_review(c, g, sql, digest(sql))["status"])
            sql.write_text('SELECT x FROM t')
            self.assertEqual("unspecified_by_query", order_review(c, g, sql, digest(sql))["row_order"])

    def test_separate_capture_roots_exact_values_overflow_and_wrong_oracles(self):
        with tempfile.TemporaryDirectory() as temporary:
            cpu, gpu = [Path(temporary) / e for e in ("cpu", "gpu")]
            for p in (cpu, gpu):
                (p / "batches/00001/captures").mkdir(parents=True)
                capture = p / "batches/00001/captures/1.jsonl"
                capture.write_text('[["integer",1]]\n')
                manifest = dict(dataset="nds", engine=p.name, selected=["1"], data="fixed-data",
                                query_sha256={"1": "adapted-hash"}, upstream_revision="fixed-revision", sql_configuration={})
                atomic_json(p / "manifest.json", manifest)
                record = dict(id="1", batch=1, status="ok", row_count=1, schema={"type": "struct"},
                              capture="captures/1.jsonl", capture_sha256=digest(capture))
                (p / "results.jsonl").write_text(json.dumps(record) + "\n")
            self.assertEqual({"match": 1}, compare(cpu, gpu)["counts"])
            for root in (cpu, gpu):
                m = json.loads((root / "manifest.json").read_text())
                m["data"] = dict(root=str(root / "data"), bytes=10,
                                 files=[dict(path="t/part.parquet", size=10, sha256="same-bytes")])
                atomic_json(root / "manifest.json", m)
            self.assertEqual({"match": 1}, compare(cpu, gpu)["counts"])
            changed = json.loads((gpu / "manifest.json").read_text())
            changed["data"]["files"][0]["sha256"] = "different-bytes"
            atomic_json(gpu / "manifest.json", changed)
            with self.assertRaisesRegex(ValueError, "differ: data"):
                compare(cpu, gpu)
            changed["data"]["files"][0]["sha256"] = "same-bytes"
            atomic_json(gpu / "manifest.json", changed)
            capture = gpu / "batches/00001/captures/1.jsonl"
            capture.write_text('[["integer",2]]\n')
            record["capture_sha256"] = digest(capture)
            (gpu / "results.jsonl").write_text(json.dumps(record) + "\n")
            self.assertEqual({"mismatch": 1}, compare(cpu, gpu)["counts"])
            record.pop("capture")
            record["capture_status"] = "capture_limit"
            (gpu / "results.jsonl").write_text(json.dumps(record) + "\n")
            result = compare(cpu, gpu)
            self.assertEqual({"not_validated": 1}, result["counts"])
            self.assertEqual("complete_capture_unavailable", result["queries"][0]["reason"])
            with self.assertRaisesRegex(ValueError, "CPU oracle"):
                compare(gpu, gpu)
            manifest["engine"] = "gpu"
            manifest["query_sha256"] = {"1": "other-sql"}
            atomic_json(gpu / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "query_sha256"):
                compare(cpu, gpu)


if __name__ == "__main__":
    unittest.main()
