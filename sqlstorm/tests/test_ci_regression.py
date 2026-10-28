# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sqlstorm.ci_regression import remote_command


class RemoteModes(unittest.TestCase):
    def test_remote_modes_preserve_arguments_and_gpu_never_launches_verification(self):
        with tempfile.TemporaryDirectory(prefix="storm modes ") as temporary:
            root = Path(temporary)
            source = root / "source/sqlstorm"
            source.mkdir(parents=True)
            (source / "__init__.py").touch()
            for module in ("verify", "regression"):
                (source / (module + ".py")).write_text("import json,sys; print(json.dumps([__name__,sys.argv]))")
            injected = root / "injected"
            data = "literal ' $(touch " + str(injected) + ")"
            for mode, expected in [("cpu-baseline", 2), ("gpu-regression", 1)]:
                command = remote_command(str(root), mode, data, "saved cpu", "spark path", sys.executable, "java path", 60)
                result = subprocess.run(["bash", "-c", command], cwd=root, text=True, capture_output=True, check=True)
                lines = [json.loads(line) for line in result.stdout.splitlines()]
                self.assertEqual(expected, len(lines))
                args = lines[-1][1]
                self.assertEqual(mode, args[args.index("--mode") + 1])
                self.assertEqual(data, args[args.index("--data") + 1])
                self.assertEqual(mode == "gpu-regression", "--plugin-jar" in args)
                self.assertFalse(injected.exists())


if __name__ == "__main__":
    unittest.main()
