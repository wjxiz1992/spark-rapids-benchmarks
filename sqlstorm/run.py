# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU/GPU pilot. Only local Spark masters and small local Parquet inputs.

Exit 0: selected deterministic queries matched, with requested GPU evidence.
Exit 1: completed selected work contains query/comparison/coverage failures.
Exit 2: incomplete campaign, changed inputs, recording or setup failure.
The full corpus and unscheduled queries remain explicit in the bundle.
"""
import argparse
import collections
import datetime
import html
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

try:
    from .common import POLICY, atomic_json, compare_results, digest, event
except ImportError:
    from common import POLICY, atomic_json, compare_results, digest, event

TABLES = ["call_center", "catalog_page", "catalog_returns", "catalog_sales", "customer",
          "customer_address", "customer_demographics", "date_dim", "household_demographics",
          "income_band", "inventory", "item", "promotion", "reason", "ship_mode", "store",
          "store_returns", "store_sales", "time_dim", "warehouse", "web_page", "web_returns",
          "web_sales", "web_site"]
SQL_CONF = {"spark.sql.session.timeZone": "UTC", "spark.sql.ansi.enabled": "false",
            "spark.sql.adaptive.enabled": "true", "spark.sql.shuffle.partitions": "8"}


def dataset_identity(root, max_bytes):
    root = Path(root).resolve()
    missing = [table for table in TABLES if not (root / table).is_dir()]
    if missing:
        raise ValueError("Missing TPC-DS tables: " + ",".join(missing))
    files, total = [], 0
    for table in TABLES:
        paths = sorted((root / table).rglob("*.parquet"))
        if not paths:
            raise ValueError("No Parquet files in " + table)
        for path in paths:
            if not path.resolve().is_relative_to(root) or path.is_symlink():
                raise ValueError("Dataset contains an external/symlinked file")
            total += path.stat().st_size
            files.append(path)
            if total > max_bytes or len(files) > 10000:
                raise ValueError("Pilot dataset exceeds size/file budget; 3 TB runs are not enabled")
    return dict(root=str(root), bytes=total, files=[
        dict(path=str(p.relative_to(root)), size=p.stat().st_size, sha256=digest(p)) for p in files])


def stop_process(process):
    """Only terminate the process group created for this local Spark worker."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=10)
        return
    # A dead launcher can leave a JVM/Python descendant in its original group.
    # Reap the launcher, but check and stop the owned group as well.
    until = time.monotonic() + 10
    while time.monotonic() < until:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def execute_engine(args, root, selected, engine, deadline):
    journal = root / (engine + ".jsonl")
    journal.touch(exist_ok=False)
    remaining = list(selected)
    outcomes, ready = {}, []
    launches = 0
    with journal.open(encoding="utf-8") as reader:
        while remaining and time.monotonic() < deadline:
            launches += 1
            jobs = [dict(id=query_id, attempt_id=uuid.uuid4().hex) for query_id in remaining]
            job = dict(output=str(root), engine=engine, data=str(args.data), tables=TABLES,
                       queries=jobs, max_rows=args.max_rows, capture_bytes=args.capture_mib * 1024 ** 2,
                       query_timeout=args.query_timeout, sql_configuration=list(SQL_CONF))
            job_path = root / (engine + "-job-" + str(launches) + ".json")
            atomic_json(job_path, job)
            command = [str(args.spark_home / "bin/spark-submit"), "--master", "local[" + str(args.cores) + "]",
                       "--driver-memory", "4g", "--conf", "spark.ui.enabled=false",
                       "--conf", "spark.driver.maxResultSize=512m",
                       "--conf", "spark.driver.extraJavaOptions=-XX:ActiveProcessorCount=" + str(args.cores),
                       "--conf", "spark.local.dir=" + str(root / "scratch" / engine),
                       "--conf", "spark.sql.warehouse.dir=" + str(root / "scratch" / (engine + "-warehouse"))]
            for key, value in SQL_CONF.items():
                command += ["--conf", key + "=" + value]
            if engine == "gpu":
                command += ["--jars", str(args.plugin_jar), "--conf", "spark.plugins=com.nvidia.spark.SQLPlugin",
                            "--conf", "spark.rapids.sql.enabled=true",
                            "--conf", "spark.rapids.memory.gpu.allocFraction=0.3",
                            "--conf", "spark.rapids.memory.gpu.maxAllocFraction=0.3",
                            "--conf", "spark.rapids.memory.gpu.minAllocFraction=0",
                            "--conf", "spark.rapids.sql.concurrentGpuTasks=1"]
            else:
                command += ["--conf", "spark.plugins=", "--conf", "spark.rapids.sql.enabled=false"]
            command += [str(root / "inputs/source/worker.py"), "--job", str(job_path)]
            env = dict(os.environ, PYSPARK_PYTHON=str(args.python), PYSPARK_DRIVER_PYTHON=str(args.python),
                       SPARK_LOCAL_IP="127.0.0.1", TZ="UTC", CUDA_VISIBLE_DEVICES=args.gpu)
            # Inherited submit arguments must not inject RAPIDS into the CPU reference.
            env.pop("PYSPARK_SUBMIT_ARGS", None)
            launch_time = time.monotonic()
            active, active_since, worker_ready = None, None, False
            shutdown_since = None
            reason = None
            attempt_ids = {j["attempt_id"]: j["id"] for j in jobs}

            def read_events():
                nonlocal active, active_since, worker_ready, shutdown_since
                while True:
                    offset = reader.tell()
                    line = reader.readline()
                    if not line or not line.endswith("\n"):
                        reader.seek(offset)
                        break
                    item = json.loads(line)
                    if item["engine"] != engine:
                        raise ValueError("Unexpected journal engine")
                    if item["type"] == "ready":
                        ready.append(item)
                        worker_ready = True
                    elif item["type"] == "started":
                        if active is not None or attempt_ids.get(item["attempt_id"]) != item["id"]:
                            raise ValueError("Unexpected query start")
                        active, active_since = item, time.monotonic()
                    elif item["type"] == "finished":
                        if (active is None or active["attempt_id"] != item["attempt_id"]
                                or item["id"] in outcomes):
                            raise ValueError("Unexpected/duplicate terminal record")
                        outcomes[item["id"]] = item
                        active, active_since = None, None
                        shutdown_since = time.monotonic()
                    else:
                        raise ValueError("Unknown journal event")

            with (root / "logs" / (engine + "-" + str(launches) + ".log")).open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=True)
                try:
                    while True:
                        read_events()
                        now = time.monotonic()
                        if process.poll() is not None:
                            read_events()
                            break
                        if now >= deadline:
                            reason = "whole_run_timeout"
                        elif not worker_ready and now - launch_time > args.startup_timeout:
                            reason = "startup_timeout"
                        elif active is not None and now - active_since > args.query_timeout:
                            reason = "query_timeout"
                        elif active is None and shutdown_since and now - shutdown_since > 30:
                            reason = "worker_shutdown_or_idle_timeout"
                        if reason:
                            stop_process(process)
                            read_events()
                            break
                        time.sleep(0.2)
                finally:
                    stop_process(process)
            if active is not None:
                item = dict(active, type="finished", status="timeout" if reason else "crash",
                            elapsed_seconds=time.monotonic() - active_since,
                            error=reason or "Worker exited without terminal result", row_count=0)
                event(journal, item)
                outcomes[item["id"]] = item
                reader.seek(0, os.SEEK_END)
            remaining = [query_id for query_id in selected if query_id not in outcomes]
            if not worker_ready or reason == "whole_run_timeout":
                return outcomes, ready, reason or "worker_setup_failed"
            if not remaining:
                return outcomes, ready, None if process.returncode == 0 else "worker_shutdown_failed"
            if launches >= len(selected) + 1:
                return outcomes, ready, "worker_restart_limit"
    return outcomes, ready, "whole_run_timeout" if remaining else None


def coverage_status(query):
    if query["comparison"]["status"] != "match":
        return "not_validated"
    if query["comparison"]["row_count"] == 0:
        return "empty_result"
    return "gpu_metrics_observed" if query.get("gpu_active_operators") else "no_gpu_evidence"


def pilot_passes(queries):
    return (bool(queries) and all(q["comparison"]["status"] == "match"
            and coverage_status(q) != "no_gpu_evidence" for q in queries)
            and any(coverage_status(q) == "gpu_metrics_observed" for q in queries))


def write_report(root, summary):
    atomic_json(root / "summary.json", summary)
    suite = ET.Element("testsuite", name="SQLStorm Spark pilot", tests=str(len(summary["queries"])))
    for query in summary["queries"]:
        case = ET.SubElement(suite, "testcase", classname="sqlstorm.tpcds", name=query["id"])
        coverage = coverage_status(query)
        ET.SubElement(case, "system-out").text = "GPU coverage: " + coverage
        if query["comparison"]["status"] != "match" or coverage == "no_gpu_evidence":
            ET.SubElement(case, "failure", message=query["comparison"]["status"] + ": " + coverage).text = json.dumps(query)
    if summary["queries"] and not any(coverage_status(q) == "gpu_metrics_observed" for q in summary["queries"]):
        case = ET.SubElement(suite, "testcase", classname="sqlstorm.coverage", name="nonempty_gpu_work")
        ET.SubElement(case, "failure", message="No nonempty GPU query with positive output metrics")
    if summary["lifecycle_state"] != "completed":
        case = ET.SubElement(suite, "testcase", classname="sqlstorm.campaign", name="completion")
        ET.SubElement(case, "failure", message="incomplete").text = str(summary.get("error", summary.get("engine_errors", "Run is not finalized")))
    suite.set("tests", str(len(suite)))
    suite.set("failures", str(sum(case.find("failure") is not None for case in suite)))
    ET.ElementTree(suite).write(root / "junit.xml", encoding="utf-8", xml_declaration=True)
    rows = []
    for query in summary["queries"]:
        cells = [query["id"], query["cpu"].get("status", "not_run"),
                 query["gpu"].get("status", "not_run"), query["comparison"]["status"],
                 str(query["comparison"].get("row_count", "")),
                 coverage_status(query)]
        rows.append("<tr>" + "".join("<td>" + html.escape(cell) + "</td>" for cell in cells) + "</tr>")
    (root / "report.html").write_text("<!doctype html><meta charset=utf-8><title>SQLStorm Spark pilot</title>"
        "<style>body{font:16px system-ui;margin:3em;max-width:1000px}td,th{padding:.5em;border:1px solid #ccc}table{border-collapse:collapse}</style>"
        "<h1>SQLStorm Spark pilot</h1><p>" + html.escape(summary["lifecycle_state"]) + "</p>"
        "<p>Selected queries: " + str(summary["scheduled_queries"]) + " / original corpus: " + str(summary["corpus_count"]) + "</p>"
        "<p>Unordered exact typed values. Timings are diagnostic. GPU evidence is a consumed Spark plan.</p>"
        "<table><tr><th>Original ID</th><th>CPU</th><th>GPU</th><th>Comparison</th><th>Rows</th><th>GPU evidence</th></tr>"
        + "".join(rows) + "</table><p><a href=summary.json>Complete JSON report</a></p>", encoding="utf-8")


def main(argv=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Campaign received signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-label", required=True, help="Observed input identity; not a TPC score")
    parser.add_argument("--spark-home", type=Path, required=True)
    parser.add_argument("--plugin-jar", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--whole-run-timeout", type=float, default=1200)
    parser.add_argument("--query-timeout", type=float, default=60)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--capture-mib", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--max-data-mib", type=int, default=2048)
    args = parser.parse_args(argv)
    if not (1 <= args.cores <= 4 and 0 < args.whole_run_timeout <= 1800
            and 0 < args.query_timeout <= 120 and 0 < args.startup_timeout <= 300
            and 0 < args.capture_mib <= 64 and 0 < args.max_rows <= 100000
            and 0 < args.max_data_mib <= 4096 and re.fullmatch(r"\d+", args.gpu)):
        parser.error("Pilot resource bounds exceeded")
    for name in ("bundle", "data", "spark_home", "plugin_jar", "python", "output"):
        setattr(args, name, getattr(args, name).resolve())
    bundle = json.loads((args.bundle / "bundle.json").read_text())
    selected = bundle["selected"]
    corpus = {q["id"]: q for q in bundle["corpus"]}
    if (bundle["schema_version"] != "sqlstorm.spark.bundle.v1" or len(corpus) != bundle["corpus_count"]
            or not selected or len(selected) > 50 or len(selected) != len(set(selected))):
        parser.error("Invalid or oversized pilot bundle")
    for query_id in selected:
        if not re.fullmatch(r"\d+", query_id) or query_id not in corpus:
            parser.error("Invalid query identity")
        if corpus[query_id]["review_flags"]:
            parser.error("Selected query requires determinism review: " + query_id)
        if digest(args.bundle / "queries" / (query_id + ".sql")) != corpus[query_id]["sha256"]:
            parser.error("Selected SQL hash differs")
    if not args.plugin_jar.is_file() or not (args.spark_home / "bin/spark-submit").is_file():
        parser.error("Missing Spark or plugin runtime")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    for directory in ("inputs", "inputs/source", "logs", "plans", "captures", "scratch"):
        (root / directory).mkdir(exist_ok=True)
    began = time.monotonic()
    deadline = began + args.whole_run_timeout
    summary = dict(schema_version="sqlstorm.spark.run.v1", campaign_id=str(uuid.uuid4()),
                   lifecycle_state="incomplete", policy=POLICY, dataset_label=args.dataset_label,
                   corpus_count=bundle["corpus_count"], scheduled_queries=len(selected), queries=[])
    write_report(root, summary)
    try:
        shutil.copytree(args.bundle, root / "inputs/bundle")
        for name in ("common.py", "worker.py", "run.py", "bundle.py"):
            shutil.copyfile(Path(__file__).with_name(name), root / "inputs/source" / name)
        data = dataset_identity(args.data, args.max_data_mib * 1024 ** 2)
        provenance = args.data / "dataset.json"
        if provenance.is_file():
            shutil.copyfile(provenance, root / "inputs/dataset-provenance.json")
            data["declared_provenance_sha256"] = digest(provenance)
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.free,driver_version",
                                             "--format=csv,noheader,nounits"], text=True)
        gpu_lines = gpu_info.strip().splitlines()
        if int(args.gpu) >= len(gpu_lines) or int(gpu_lines[int(args.gpu)].split(",")[2]) < 15360:
            raise RuntimeError("Selected GPU needs at least 15 GiB free VRAM")
        runtime = dict(plugin_jar=str(args.plugin_jar), plugin_sha256=digest(args.plugin_jar),
                       spark_home=str(args.spark_home), python=str(args.python), gpu=gpu_info,
                       host=os.uname().nodename, sql_configuration=SQL_CONF,
                       spark_jars_sha256={p.name: digest(p) for p in sorted((args.spark_home / "jars").glob("*.jar"))},
                       spark_conf_sha256={p.name: digest(p) for p in Path(os.environ.get(
                           "SPARK_CONF_DIR", str(args.spark_home / "conf"))).glob("*") if p.is_file()},
                       source_sha256={p.name: digest(p) for p in (root / "inputs/source").iterdir()})
        atomic_json(root / "manifest.json", dict(summary, dataset=data, runtime=runtime,
                    limits={key: value for key, value in vars(args).items() if not isinstance(value, Path)},
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    bundle_sha256=digest(root / "inputs/bundle/bundle.json")))
        cpu_began = time.monotonic()
        cpu, cpu_ready, cpu_error = execute_engine(args, root, selected, "cpu", deadline)
        cpu_seconds = time.monotonic() - cpu_began
        gpu_began = time.monotonic()
        gpu, gpu_ready, gpu_error = execute_engine(args, root, selected, "gpu", deadline)
        gpu_seconds = time.monotonic() - gpu_began
        for query_id in selected:
            reference, candidate = cpu.get(query_id, {}), gpu.get(query_id, {})
            summary["queries"].append(dict(id=query_id, sql_sha256=corpus[query_id]["sha256"],
                cpu=reference, gpu=candidate, gpu_operators=candidate.get("gpu_operators", []),
                gpu_active_operators=candidate.get("gpu_active_operators", []),
                comparison=compare_results(reference, candidate, root)))
        observed_data = dataset_identity(args.data, args.max_data_mib * 1024 ** 2)
        if provenance.is_file():
            observed_data["declared_provenance_sha256"] = digest(provenance)
        unchanged = (observed_data == data
                     and digest(args.plugin_jar) == runtime["plugin_sha256"])
        summary.update(engine_errors={"cpu": cpu_error, "gpu": gpu_error},
                       ready={"cpu": cpu_ready, "gpu": gpu_ready}, inputs_unchanged=unchanged,
                       comparison_counts=dict(collections.Counter(q["comparison"]["status"] for q in summary["queries"])),
                       gpu_plan_queries=sum(bool(q["gpu_operators"]) for q in summary["queries"]),
                       coverage_counts=dict(collections.Counter(coverage_status(q) for q in summary["queries"])),
                       timings=dict(cpu_batch_seconds=cpu_seconds, gpu_batch_seconds=gpu_seconds,
                                    total_seconds=time.monotonic() - began),
                       lifecycle_state="completed" if not cpu_error and not gpu_error and unchanged else "incomplete")
        if not unchanged:
            for query in summary["queries"]:
                query["comparison"] = dict(status="not_validated", reason="inputs_changed")
            summary["comparison_counts"] = {"not_validated": len(summary["queries"])}
        write_report(root, summary)
        print(json.dumps({k: summary[k] for k in ("lifecycle_state", "scheduled_queries", "comparison_counts", "gpu_plan_queries", "timings")}))
        if summary["lifecycle_state"] != "completed":
            return 2
        return 0 if pilot_passes(summary["queries"]) else 1
    except BaseException as exc:
        summary["error"] = repr(exc)
        write_report(root, summary)
        if isinstance(exc, KeyboardInterrupt):
            return 130
        raise


if __name__ == "__main__":
    raise SystemExit(main())
