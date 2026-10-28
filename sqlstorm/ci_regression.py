# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the two explicit SQLStorm modes under Jenkins's SPARK2A_CLUSTER lock."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile


def remote_command(root, mode, data, baseline, spark, python, java, budget):
    environment = ["env", "JAVA_HOME=" + java, "SPARK_CONF_DIR=" + root + "/spark-conf",
                   "PYTHONPATH=" + root + "/source:" + root + "/python-tools"]
    run = environment + [python, "-m", "sqlstorm.regression", "--mode", mode,
          "--bundle", root + "/bundle", "--data", data, "--baseline", baseline,
          "--spark-home", spark, "--python", python, "--output", root + "/run",
          "--budget", str(budget), "--capture-mib", "32"]
    if mode == "gpu-regression":
        run += ["--plugin-jar", root + "/plugin.jar"]
    commands = []
    if mode == "cpu-baseline":
        commands.append(shlex.join(environment + [python, "-m", "sqlstorm.verify", "--bundle", root + "/bundle",
                                                  "--data", data, "--capture-mib", "32"]))
    commands.append(shlex.join(run))
    return shlex.join(["timeout", "--signal=TERM", "--kill-after=30s", str(budget + 900) + "s",
                       "bash", "-c", " && ".join(commands)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ssh-key", "bundle", "dependencies", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("run-id", "data", "baseline", "spark-home"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=("cpu-baseline", "gpu-regression"), required=True)
    parser.add_argument("--plugin-jar", type=Path)
    parser.add_argument("--python", required=True, help="Target interpreter matching the staged dependency wheels")
    parser.add_argument("--java-home", default="/usr/lib/jvm/java-17-openjdk-amd64")
    parser.add_argument("--budget", type=int, default=1200)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", args.run_id) or not 60 <= args.budget <= 1800:
        parser.error("Unsafe run ID or excessive budget")
    if args.mode == "gpu-regression" and (not args.plugin_jar or not args.plugin_jar.is_file()):
        parser.error("GPU mode requires a candidate artifact")
    if not Path(args.baseline).is_absolute():
        parser.error("Select an absolute durable CPU baseline directory")
    args.output.mkdir(parents=True, exist_ok=False)
    root = "/tmp/sqlstorm-cudf-spark/" + args.run_id
    target = "rladmin@rl-r7525-d32-u13.raplab.nvidia.com"
    options = ["-i", str(args.ssh_key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
               "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
               "-o", "StrictHostKeyChecking=accept-new"]
    ssh, scp = ["ssh", *options, target], ["scp", *options]
    archive = args.output / "inputs.tar.gz"
    source = Path(__file__).resolve().parent
    with tarfile.open(archive, "w:gz") as out:
        for path in sorted(source.glob("*.py")):
            out.add(path, arcname="source/sqlstorm/" + path.name)
        for folder, name in [(args.bundle, "bundle"), (args.dependencies, "python-tools")]:
            out.add(folder, arcname=name, filter=lambda info: None if "__pycache__" in info.name else info)
    subprocess.run(ssh + ["mkdir -p /tmp/sqlstorm-cudf-spark && mkdir " + shlex.quote(root)], check=True, timeout=45)
    subprocess.run(scp + [str(archive), target + ":" + root + "/inputs.tar.gz"], check=True, timeout=180)
    if args.mode == "gpu-regression":
        subprocess.run(scp + [str(args.plugin_jar), target + ":" + root + "/plugin.jar"], check=True, timeout=240)
    subprocess.run(ssh + ["cd " + shlex.quote(root) + " && tar -xzf inputs.tar.gz && mkdir spark-conf"], check=True, timeout=60)
    print("Retained remote evidence: " + root + "; CPU baseline: " + args.baseline, flush=True)
    status = 2
    try:
        status = subprocess.run(ssh + [remote_command(root, args.mode, args.data, args.baseline,
            args.spark_home, args.python, args.java_home, args.budget)], timeout=args.budget + 1050).returncode
    finally:
        pack = ("import pathlib,tarfile; r=pathlib.Path(" + repr(root) + "); "
                "a=tarfile.open(r/'evidence.tar.gz','w:gz'); "
                "[a.add(r/n,arcname=n) for n in ('run','bundle') if (r/n).exists()]; a.close()")
        subprocess.run(ssh + [shlex.join([args.python, "-c", pack])], check=True, timeout=180)
        subprocess.run(scp + [target + ":" + root + "/evidence.tar.gz", str(args.output / "evidence.tar.gz")], check=True, timeout=180)
        for name in ("summary.json", "manifest.json", "junit.xml"):
            subprocess.run(scp + [target + ":" + root + "/run/" + name, str(args.output / name)], timeout=60, check=False)
        (args.output / "remote.json").write_text(json.dumps(dict(root=root, baseline=args.baseline, mode=args.mode, exit_code=status)) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
