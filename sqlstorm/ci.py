# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage one bounded pilot on a Spark2a worker while Jenkins holds SPARK2A_CLUSTER.

Credentials stay in the Jenkins agent. No cluster installation, Spark service,
existing NDS dataset or application is changed. The remote run and evidence are
retained under a unique directory; only this pilot's local Spark processes stop.
"""
import argparse
from pathlib import Path
import re
import shlex
import subprocess
import tarfile


def remote_command(root, data, spark_home, python, java_home, label, query_timeout, budget):
    args = [python, root + "/source/sqlstorm/run.py", "--bundle", root + "/bundle",
            "--data", data, "--dataset-label", label, "--spark-home", spark_home,
            "--plugin-jar", root + "/plugin.jar", "--python", python, "--output", root + "/run",
            "--whole-run-timeout", str(budget), "--query-timeout", str(query_timeout)]
    environment = ["env", "JAVA_HOME=" + java_home, "SPARK_CONF_DIR=" + root + "/spark-conf"]
    return shlex.join(["timeout", "--signal=TERM", "--kill-after=20s", str(budget + 60) + "s"]
                      + environment + args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--plugin-jar", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--spark-home", default="/opt/spark/spark-3.5.7-bin-hadoop3")
    parser.add_argument("--python", default="/usr/bin/python3")
    parser.add_argument("--java-home", default="/usr/lib/jvm/java-17-openjdk-amd64")
    parser.add_argument("--query-timeout", type=int, default=60)
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", args.run_id):
        parser.error("Unsafe run ID")
    if not (0 < args.budget <= 1800 and 0 < args.query_timeout <= 120):
        parser.error("Pilot deadline bounds exceeded")
    args.output.mkdir(parents=True, exist_ok=False)
    root = "/tmp/sqlstorm-cudf-spark/" + args.run_id
    target = "rladmin@rl-r7525-d32-u13.raplab.nvidia.com"
    options = ["-i", str(args.ssh_key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
               "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
               "-o", "StrictHostKeyChecking=accept-new"]
    ssh = ["ssh", *options, target]
    scp = ["scp", *options]
    archive = args.output / "inputs.tar.gz"
    source = Path(__file__).resolve().parent
    with tarfile.open(archive, "w:gz") as out:
        for name in ("common.py", "worker.py", "run.py", "bundle.py"):
            out.add(source / name, arcname="source/sqlstorm/" + name)
        for path in sorted(args.bundle.rglob("*")):
            if path.is_file():
                out.add(path, arcname="bundle/" + str(path.relative_to(args.bundle)))
    subprocess.run(ssh + ["mkdir -p /tmp/sqlstorm-cudf-spark && mkdir " + shlex.quote(root)], check=True, timeout=45)
    subprocess.run(scp + [str(archive), target + ":" + root + "/inputs.tar.gz"], check=True, timeout=120)
    subprocess.run(scp + [str(args.plugin_jar), target + ":" + root + "/plugin.jar"], check=True, timeout=180)
    subprocess.run(ssh + ["cd " + shlex.quote(root) + " && tar -xzf inputs.tar.gz && mkdir spark-conf"],
                   check=True, timeout=45)
    print("Retained remote evidence directory: " + root, flush=True)
    status = 2
    try:
        status = subprocess.run(ssh + [remote_command(root, args.data, args.spark_home, args.python,
            args.java_home, args.dataset_label, args.query_timeout, args.budget)],
            timeout=args.budget + 150).returncode
    finally:
        # The remote timeout sends SIGTERM to the supervisor, whose finally block
        # stops its own isolated Spark process group before preserving the report.
        subprocess.run(ssh + ["cd " + shlex.quote(root) + " && tar -czf evidence.tar.gz run"],
                       check=True, timeout=180)
        subprocess.run(scp + [target + ":" + root + "/evidence.tar.gz", str(args.output / "evidence.tar.gz")],
                       check=True, timeout=180)
        for filename in ("summary.json", "manifest.json", "junit.xml", "report.html"):
            subprocess.run(scp + [target + ":" + root + "/run/" + filename,
                                   str(args.output / filename)], check=True, timeout=60)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
