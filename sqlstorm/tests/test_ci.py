# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from sqlstorm.ci import remote_command
from sqlstorm.run import TABLES, dataset_identity


class CIResources(unittest.TestCase):
    def test_remote_parameters_cannot_inject_shell_commands(self):
        with tempfile.TemporaryDirectory(prefix="storm shell ") as temporary:
            root = Path(temporary)
            (root / "source/sqlstorm").mkdir(parents=True)
            script = root / "source/sqlstorm/run.py"
            script.write_text("import json,sys;print(json.dumps(sys.argv))")
            sentinel = root / "injected"
            label = "label ' $(touch " + str(sentinel) + "); echo bad"
            command = remote_command(str(root), "path with spaces", "spark path", sys.executable,
                                     "java path", label, 60, 120)
            run = subprocess.run(["bash", "-c", command], capture_output=True, text=True, check=True)
            arguments = json.loads(run.stdout)
            self.assertEqual(label, arguments[arguments.index("--dataset-label") + 1])
            self.assertEqual("path with spaces", arguments[arguments.index("--data") + 1])
            self.assertFalse(sentinel.exists())

    def test_dataset_budget_rejects_before_hashing_large_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for table in TABLES:
                (root / table).mkdir()
                (root / table / "part.parquet").write_bytes(b"data")
            with self.assertRaisesRegex(ValueError, "size/file budget"):
                dataset_identity(root, max_bytes=10)

    def test_symlinked_inputs_do_not_escape_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / "outside"
            outside.write_bytes(b"external")
            for table in TABLES:
                (root / table).mkdir()
                (root / table / "part.parquet").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlinked"):
                dataset_identity(root, max_bytes=1000)


if __name__ == "__main__":
    unittest.main()
