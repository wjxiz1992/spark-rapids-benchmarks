# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest

from sqlstorm.common import POLICY, atomic_json, digest
from sqlstorm.regression import (REFERENCE_VERSION, TIMING_POLICY, load_reference,
                                make_report, validate_equivalence)


class References(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.capture = self.root / "captures/cpu-7.jsonl"
        self.capture.parent.mkdir()
        self.capture.write_text('[["integer",1]]\n')
        self.manifest = dict(dataset="nds", selected=["7"])
        self.hashes = {"7": "query-hash"}
        self.data = dict(bytes=1, files=[dict(path="t/a.parquet", size=1, sha256="data-hash")])
        self.resources = dict(cores=4, driver_memory="4g")
        self.reference = dict(schema_version=REFERENCE_VERSION, **self.manifest,
                              query_sha256=self.hashes, data=self.data, resources=self.resources,
                              sql_configuration={"spark.sql.session.timeZone": "UTC", "spark.sql.ansi.enabled": "false",
                                                 "spark.sql.adaptive.enabled": "true", "spark.sql.shuffle.partitions": "8"},
                              comparison_policy=POLICY, timing_policy=TIMING_POLICY, spark_version="4.0.4",
                              queries=[dict(id="7", status="ok", capture="captures/cpu-7.jsonl", row_count=1,
                                            capture_sha256=digest(self.capture))])
        atomic_json(self.root / "baseline.json", self.reference)

    def test_existing_reference_can_be_selected_without_running_cpu(self):
        reference, records = load_reference(self.root, self.manifest, self.hashes, self.data, self.resources)
        self.assertEqual("4.0.4", reference["spark_version"])
        self.assertEqual(1, records["7"]["row_count"])

    def test_changed_query_data_or_resource_configuration_is_rejected(self):
        for hashes, data, resources in [({"7": "changed"}, self.data, self.resources),
                                       (self.hashes, {"bytes": 2}, self.resources),
                                       (self.hashes, self.data, {"cores": 2})]:
            with self.subTest(hashes=hashes, data=data, resources=resources), self.assertRaises(ValueError):
                load_reference(self.root, self.manifest, hashes, data, resources)

    def test_corrupt_complete_capture_is_not_a_reference(self):
        self.capture.write_text('[["integer",2]]\n')
        with self.assertRaisesRegex(ValueError, "capture changed"):
            load_reference(self.root, self.manifest, self.hashes, self.data, self.resources)

    def test_missing_query_record_cannot_be_hidden_by_successful_queries(self):
        self.reference["queries"] = []
        atomic_json(self.root / "baseline.json", self.reference)
        with self.assertRaisesRegex(ValueError, "missing/duplicate"):
            load_reference(self.root, self.manifest, self.hashes, self.data, self.resources)

    def test_identity_claim_cannot_cover_changed_sql(self):
        manifest = dict(self.manifest, adaptations=[dict(id="7", original_sha256="old", adapted_sha256="new")])
        proof = dict(success=True, dataset="nds", selected=["7"], data=self.data,
                     queries=[dict(id="7", status="identity", original_sha256="old", adapted_sha256="new")])
        with self.assertRaisesRegex(ValueError, "Changed/unvalidated"):
            validate_equivalence(proof, manifest, self.data)

    def test_empty_result_is_a_valid_correctness_match_without_plan_coverage(self):
        manifest = dict(mode="gpu-regression", dataset="nds", selected=["7"], corpus_count=15242, baseline="saved-cpu")
        self.assertTrue(make_report(self.root, manifest, [dict(id="7", status="match", row_count=0)], "complete"))
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual("PASS", summary["quality_status"])
        self.assertFalse(summary["gpu_coverage_gate"])

    def test_missing_reference_and_unexecuted_queries_do_not_pass(self):
        manifest = dict(mode="gpu-regression", dataset="nds", selected=["7"], corpus_count=15242, baseline="saved-cpu")
        self.assertFalse(make_report(self.root, manifest, [dict(id="7", status="not_validated")], "complete"))
        self.assertFalse(make_report(self.root, manifest, [], "incomplete", "startup failed"))


if __name__ == "__main__":
    unittest.main()
