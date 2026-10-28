# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sqlstorm.common import digest
from sqlstorm.regression import TIMING_POLICY, main


class Modes(unittest.TestCase):
    def test_manual_cpu_then_gpu_only_and_missing_baseline_never_runs_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bundle/queries").mkdir(parents=True)
            (root / "bundle/queries/1.sql").write_text("SELECT 1")
            (root / "spark/bin").mkdir(parents=True)
            (root / "spark/bin/spark-submit").touch()
            (root / "python").touch()
            (root / "plugin.jar").write_bytes(b"test artifact")
            bundle = dict(dataset="nds", selected=["1"], corpus_count=15242, upstream_revision="fixed")
            common = ["--bundle", str(root / "bundle"), "--data", str(root / "data"),
                      "--spark-home", str(root / "spark"), "--python", str(root / "python"),
                      "--baseline", str(root / "reference")]
            engines = []

            def execute(args, output, selected, engine, deadline):
                engines.append(engine)
                path = output / "captures" / (engine + ".jsonl")
                path.write_text('[["integer",1]]\n')
                record = dict(id="1", status="ok", schema={"type": "struct", "fields": []},
                              capture="captures/" + path.name, capture_sha256=digest(path), row_count=1,
                              query_seconds=2 if engine == "cpu" else 1, timing_policy=TIMING_POLICY)
                return {"1": record}, [dict(spark_version="4.0.4")], None

            with patch("sqlstorm.regression.load_bundle", return_value=(bundle, {})), \
                    patch("sqlstorm.regression.data_identity", return_value={"files": [], "bytes": 0}), \
                    patch("sqlstorm.regression.execute_engine", side_effect=execute), \
                    patch("subprocess.check_output", return_value="40000\n"):
                self.assertEqual(0, main(common + ["--mode", "cpu-baseline", "--output", str(root / "cpu")]))
                original = (root / "reference/baseline.json").read_bytes()
                self.assertEqual(0, main(common + ["--mode", "gpu-regression", "--output", str(root / "gpu"),
                                                  "--plugin-jar", str(root / "plugin.jar")]))
                self.assertEqual(["cpu", "gpu"], engines)
                self.assertEqual(original, (root / "reference/baseline.json").read_bytes())
                report = json.loads((root / "gpu/summary.json").read_text())
                self.assertEqual(2, report["queries"][0]["speedup"])
                (root / "reference/baseline.json").unlink()
                with self.assertRaises(FileNotFoundError):
                    main(common + ["--mode", "gpu-regression", "--output", str(root / "missing"),
                                   "--plugin-jar", str(root / "plugin.jar")])
                self.assertEqual(["cpu", "gpu"], engines)


if __name__ == "__main__":
    unittest.main()
