# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from sqlstorm.bundle import build, UPSTREAM_REVISION
from sqlstorm.common import digest
from sqlstorm.sweep import load_bundle, query_category, report


class ExecutionSweep(unittest.TestCase):
    def test_bundle_preserves_original_crlf_bytes_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
                "sqlstorm.bundle.DATASETS", {"nds": ("tpcds", 1)}):
            root = Path(temporary)
            source = root / "source"
            queries = source / "v1.0/tpcds/queries"
            queries.mkdir(parents=True)
            original = b"SELECT 1;\r\n"
            (queries / "1.sql").write_bytes(original)
            (queries.parent / "queries.sql").write_text("\\i v1.0/tpcds/queries/1.sql\n")
            (source / "LICENSE").write_text("MIT test fixture")
            with patch("sqlstorm.bundle.subprocess.check_output",
                       side_effect=[UPSTREAM_REVISION, ""]):
                value = build(source, root / "bundle", None)
            copied = root / "bundle/queries/1.sql"
            self.assertEqual(copied.read_bytes(), original)
            self.assertEqual(digest(copied), value["corpus"][0]["sha256"])

    def make_bundle(self, folder):
        (folder / "queries").mkdir()
        corpus = []
        for query_id, sql in [("1", "SELECT 1 LIMIT 1"), ("2", "SELECT rand()")]:
            path = folder / "queries" / (query_id + ".sql")
            path.write_text(sql)
            corpus.append(dict(id=query_id, sha256=digest(path),
                               review_flags=["row_selection_requires_review"]))
        value = dict(dataset="nds", corpus_count=2, corpus=corpus, selected=["1", "2"])
        (folder / "bundle.json").write_text(json.dumps(value))
        return value

    def test_execution_sweep_does_not_filter_review_flagged_queries(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
                "sqlstorm.sweep.DATASETS", {"nds": ("tpcds", 2)}):
            root = Path(temporary)
            self.make_bundle(root)
            self.assertEqual(load_bundle(root)[2], ["1", "2"])

    def test_modified_sql_and_unknown_query_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
                "sqlstorm.sweep.DATASETS", {"nds": ("tpcds", 2)}):
            root = Path(temporary)
            self.make_bundle(root)
            with self.assertRaisesRegex(ValueError, "Unknown query"):
                load_bundle(root, ["3"])
            (root / "queries/1.sql").write_text("SELECT 999")
            with self.assertRaisesRegex(ValueError, "changed original SQL"):
                load_bundle(root)

    def test_duplicate_corpus_entries_do_not_establish_full_coverage(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
                "sqlstorm.sweep.DATASETS", {"nds": ("tpcds", 2)}):
            root = Path(temporary)
            value = self.make_bundle(root)
            value["corpus"][1] = value["corpus"][0]
            (root / "bundle.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "duplicate corpus"):
                load_bundle(root)

    def test_successful_execution_is_never_reported_as_correctness_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = dict(dataset="nds", corpus_count=15242, selected=["1", "2"],
                            started_epoch=time.time())
            result = report(root, manifest, {"1": {"status": "ok"}}, "running")
            self.assertEqual(result["attempted"], 1)
            self.assertEqual(result["remaining"], 1)
            self.assertEqual(result["correctness_status"], "NOT_CHECKED_NO_CPU_ORACLE")

    def test_execution_errors_require_triage_instead_of_being_labelled_bugs(self):
        self.assertEqual(query_category(dict(status="error", error_type="ParseException")),
                         "sql_parse_or_analysis_error")
        self.assertEqual(query_category(dict(status="error", error_type="Py4JJavaError")),
                         "execution_error_to_triage")
        self.assertEqual(query_category(dict(status="timeout")), "timeout")


if __name__ == "__main__":
    unittest.main()
