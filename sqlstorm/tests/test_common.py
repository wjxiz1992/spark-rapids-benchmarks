# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from sqlstorm.common import compare_results, digest, encoded, error_record, scalar
from sqlstorm.bundle import review_flags
from sqlstorm.run import pilot_passes, coverage_status, write_report


class ExactResults(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.addCleanup(self.folder.cleanup)

    def capture(self, name, rows, schema="long"):
        path = self.root / name
        path.write_text("".join(encoded(row) + "\n" for row in rows))
        return dict(status="ok", attempt_id=name, capture=name, row_count=len(rows),
                    schema=schema, capture_sha256=digest(path))

    def compare(self, a, b):
        return compare_results(self.capture("cpu", a), self.capture("gpu", b), self.root)

    def test_same_count_different_values_fails(self):
        self.assertEqual("mismatch", self.compare([[1]], [[2]])["status"])

    def test_multiset_ignores_order_but_preserves_duplicate_counts(self):
        self.assertEqual("match", self.compare([[1], [2], [1]], [[2], [1], [1]])["status"])
        self.assertEqual("mismatch", self.compare([[1], [1], [2]], [[1], [2], [2]])["status"])

    def test_schema_and_capture_integrity(self):
        a, b = self.capture("a", [[1]]), self.capture("b", [[1]], "double")
        self.assertEqual("schema", compare_results(a, b, self.root)["reason"])
        (self.root / "a").write_text("[2]\n")
        self.assertEqual("not_validated", compare_results(a, b, self.root)["status"])

    def test_incomplete_or_missing_capture_never_passes(self):
        a = self.capture("a", [[1]])
        for status in ("error", "timeout", "capture_limit", "crash"):
            self.assertEqual("not_validated", compare_results(a, dict(status=status), self.root)["status"])
        b = self.capture("b", [[1]])
        (self.root / "b").unlink()
        self.assertEqual("not_validated", compare_results(a, b, self.root)["status"])

    def test_large_integers_and_decimals_remain_exact(self):
        self.assertNotEqual(scalar(2 ** 53, "long"), scalar(2 ** 53 + 1, "long"))
        self.assertNotEqual(scalar(Decimal("99999999999999999999.000000000000000001"), "decimal(38,18)"),
                            scalar(Decimal("99999999999999999999.000000000000000002"), "decimal(38,18)"))

    def test_null_and_ieee_values(self):
        self.assertEqual(scalar(float("nan"), "double"), scalar(float("nan"), "double"))
        self.assertNotEqual(scalar(-0.0, "double"), scalar(0.0, "double"))
        self.assertNotEqual(scalar(None, "double"), scalar(float("nan"), "double"))
        self.assertNotEqual(scalar(float("inf"), "double"), scalar(-float("inf"), "double"))

    def test_nested_nulls_order_and_maps(self):
        dtype = dict(type="array", elementType="integer")
        self.assertNotEqual(scalar(None, dtype), scalar([], dtype))
        self.assertNotEqual(scalar([], dtype), scalar([None], dtype))
        self.assertNotEqual(scalar([1, 2], dtype), scalar([2, 1], dtype))
        dtype = dict(type="map", keyType="string", valueType="integer")
        self.assertEqual(scalar({"a": 1, "b": 2}, dtype), scalar({"b": 2, "a": 1}, dtype))

    def test_timestamp_microseconds_and_unsupported_types(self):
        self.assertNotEqual(scalar(datetime.datetime(1970, 1, 1, microsecond=1), "timestamp"),
                            scalar(datetime.datetime(1970, 1, 1, microsecond=2), "timestamp"))
        with self.assertRaises(TypeError):
            scalar(object(), "unknown")

    def test_conservative_determinism_screen(self):
        self.assertTrue(review_flags("SELECT * FROM t LIMIT 1"))
        self.assertTrue(review_flags("SELECT * FROM t FETCH FIRST 1 ROW ONLY"))
        self.assertTrue(review_flags("SELECT * FROM t OFFSET 0 ROWS FETCH NEXT 1 ROW ONLY"))
        self.assertTrue(review_flags("SELECT row_number() OVER (ORDER BY k) FROM t"))
        self.assertTrue(review_flags("SELECT collect_list(v) FROM t"))
        self.assertEqual([], review_flags("SELECT k, sum(v) FROM t GROUP BY k"))

    def test_empty_matches_do_not_establish_gpu_coverage(self):
        empty = dict(comparison=dict(status="match", row_count=0), gpu_active_operators=[])
        real = dict(comparison=dict(status="match", row_count=1), gpu_active_operators=["GpuProjectExec"])
        fallback = dict(comparison=dict(status="match", row_count=1), gpu_active_operators=[])
        self.assertEqual("empty_result", coverage_status(empty))
        self.assertFalse(pilot_passes([empty]))
        self.assertTrue(pilot_passes([empty, real]))
        self.assertFalse(pilot_passes([real, fallback]))

    def test_completed_execution_does_not_imply_quality_pass(self):
        query = dict(id="1", cpu=dict(status="ok"), gpu=dict(status="ok"),
                     gpu_active_operators=["GpuProjectExec"],
                     comparison=dict(status="match", row_count=1))
        summary = dict(lifecycle_state="completed", queries=[query], scheduled_queries=1, corpus_count=15242)
        write_report(self.root, summary)
        self.assertEqual("PASS", json.loads((self.root / "summary.json").read_text())["quality_status"])
        query["comparison"]["status"] = "not_validated"
        write_report(self.root, summary)
        self.assertEqual("FAILURE", summary["quality_status"])
        self.assertIn("quality: FAILURE", (self.root / "report.html").read_text())
        summary["lifecycle_state"] = "incomplete"
        write_report(self.root, summary)
        self.assertEqual("INCOMPLETE", summary["quality_status"])

    def test_empty_exception_repr_does_not_lose_root_cause(self):
        class CapturedError(Exception):
            def __repr__(self):
                return "CapturedError()"
        try:
            raise CapturedError("UNSUPPORTED_SUBQUERY: original root cause\n" + "plan node\n" * 2000)
        except CapturedError as error:
            record = error_record(error)
        self.assertEqual("CapturedError", record["error_type"])
        self.assertTrue(record["error"].startswith("UNSUPPORTED_SUBQUERY: original root cause"))


if __name__ == "__main__":
    unittest.main()
