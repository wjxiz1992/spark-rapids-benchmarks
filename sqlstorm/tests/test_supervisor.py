# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real child-process deadlines without requiring Spark or a GPU."""
import argparse
from pathlib import Path
import tempfile
import time
import unittest
from sqlstorm.run import execute_engine


class Supervisor(unittest.TestCase):
    def exercise(self, source, timeout=0.25, startup=2, budget=4):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "bin").mkdir()
        (root / "logs").mkdir()
        script = root / "bin/spark-submit"
        script.write_text("#!/usr/bin/env python3\n" +
            "import sys,json,pathlib,time,os\n" +
            "job=json.loads(pathlib.Path(sys.argv[-1]).read_text())\n" +
            "p=pathlib.Path(job['output'])/(job['engine']+'.jsonl')\n" +
            "def send(v):\n with p.open('a') as f: f.write(json.dumps(dict(engine=job['engine'],**v))+'\\n'); f.flush(); os.fsync(f.fileno())\n" + source)
        script.chmod(0o755)
        args = argparse.Namespace(data=root, max_rows=10, capture_mib=1,
            query_timeout=timeout, startup_timeout=startup, spark_home=root, cores=1,
            python=Path('/usr/bin/python3'), gpu="0")
        began = time.monotonic()
        result = execute_engine(args, root, ["1"], "cpu", time.monotonic() + budget)
        self.assertLess(time.monotonic() - began, budget + 2)
        return result

    def test_native_crash_is_terminal_and_not_a_sql_success(self):
        outcomes, ready, error = self.exercise(
            "send(dict(type='ready'))\nq=job['queries'][0]\nsend(dict(type='started',**q))\nos._exit(7)\n")
        self.assertEqual("crash", outcomes["1"]["status"])
        self.assertEqual("worker_shutdown_failed", error)

    def test_hung_query_is_terminated(self):
        outcomes, ready, error = self.exercise(
            "send(dict(type='ready'))\nq=job['queries'][0]\nsend(dict(type='started',**q))\ntime.sleep(30)\n")
        self.assertEqual("timeout", outcomes["1"]["status"])
        self.assertEqual("query_timeout", outcomes["1"]["error"])

    def test_hung_startup_retains_missing_queries(self):
        outcomes, ready, error = self.exercise("time.sleep(30)\n", startup=0.25)
        self.assertEqual({}, outcomes)
        self.assertEqual("startup_timeout", error)

    def test_whole_budget_is_distinct_from_query_timeout(self):
        outcomes, ready, error = self.exercise(
            "send(dict(type='ready'))\nq=job['queries'][0]\nsend(dict(type='started',**q))\ntime.sleep(30)\n",
            timeout=30, budget=0.4)
        self.assertEqual("whole_run_timeout", error)
        self.assertEqual("timeout", outcomes["1"]["status"])


if __name__ == "__main__":
    unittest.main()
