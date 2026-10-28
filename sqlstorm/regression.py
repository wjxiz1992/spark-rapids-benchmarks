# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manually save a CPU reference, then replay only GPU queries against it."""
import argparse
import collections
import datetime
import json
from pathlib import Path
import shutil
import signal
import time
import xml.etree.ElementTree as ET

from .adapt import load_bundle
from .common import POLICY, atomic_json, compare_results, digest
from .run import SQL_CONF, TABLES, dataset_identity, execute_engine
from .sweep import NDS_H_TABLES

REFERENCE_VERSION = "sqlstorm.spark.cpu_reference.v1"
TIMING_POLICY = "planning_and_result_retrieval_excluding_capture_io_v1"


def data_identity(path, tables):
    value = dataset_identity(path, 4096 * 1024 ** 2, tables)
    return {key: value[key] for key in ("bytes", "files")}


def query_hashes(bundle, selected):
    return {qid: digest(Path(bundle) / "queries" / (qid + ".sql")) for qid in selected}


def validate_equivalence(proof, manifest, data):
    if not manifest.get("adaptations"):
        return
    if (not proof or not proof.get("success") or proof.get("data") != data
            or proof.get("dataset") != manifest["dataset"] or proof.get("selected") != manifest["selected"]):
        raise ValueError("Adapted SQL needs matching fixed-data equivalence evidence")
    records = {r["id"]: r for r in proof["queries"]}
    if len(records) != len(proof["queries"]) or set(records) != set(manifest["selected"]):
        raise ValueError("Incomplete adaptation equivalence evidence")
    for query in manifest["adaptations"]:
        record = records[query["id"]]
        if (record.get("original_sha256") != query["original_sha256"]
                or record.get("adapted_sha256") != query["adapted_sha256"]
                or record.get("status") not in ("match", "identity")
                or (record.get("status") == "identity" and query["original_sha256"] != query["adapted_sha256"])):
            raise ValueError("Changed/unvalidated adaptation: " + query["id"])


def load_reference(path, manifest, hashes, data, resources):
    path = Path(path).resolve()
    reference = json.loads((path / "baseline.json").read_text())
    expected = dict(schema_version=REFERENCE_VERSION, dataset=manifest["dataset"],
                    selected=manifest["selected"], query_sha256=hashes, data=data,
                    sql_configuration=SQL_CONF, resources=resources, comparison_policy=POLICY,
                    timing_policy=TIMING_POLICY)
    for key, value in expected.items():
        if reference.get(key) != value:
            raise ValueError("Selected CPU baseline differs: " + key)
    validate_equivalence(reference.get("adaptation_equivalence"), manifest, data)
    records = {r["id"]: r for r in reference["queries"]}
    if set(records) != set(manifest["selected"]) or len(records) != len(reference["queries"]):
        raise ValueError("CPU reference has missing/duplicate query records")
    for record in records.values():
        if record["status"] != "ok":
            continue
        original = path / record["capture"]
        capture = original.resolve()
        if not capture.is_relative_to(path) or not capture.is_file() or original.is_symlink():
            raise ValueError("CPU capture path is invalid")
        if digest(capture) != record["capture_sha256"]:
            raise ValueError("CPU reference capture changed")
    return reference, records


def publish_reference(path, root, manifest, results, ready):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    for record in results.values():
        if record["status"] == "ok":
            capture = path / record["capture"]
            capture.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / record["capture"], capture)
    versions = {r["spark_version"] for r in ready}
    if len(versions) != 1:
        raise ValueError("CPU workers did not report one Spark version")
    reference = dict(schema_version=REFERENCE_VERSION, dataset=manifest["dataset"],
                     selected=manifest["selected"], query_sha256=manifest["query_sha256"],
                     data=manifest["data"], sql_configuration=SQL_CONF,
                     resources=manifest["resources"], comparison_policy=POLICY,
                     timing_policy=TIMING_POLICY, spark_version=versions.pop(),
                     adaptation_equivalence=manifest.get("adaptation_equivalence"),
                     created_utc=manifest["started_utc"], queries=list(results.values()),
                     usable=all(r["status"] == "ok" for r in results.values()))
    # Written last: a partial directory never looks like a published baseline.
    atomic_json(path / "baseline.json", reference)
    return reference


def make_report(root, manifest, results, state, error=None):
    counts = collections.Counter(r["status"] for r in results)
    success = (state == "complete" and len(results) == len(manifest["selected"])
               and all(r["status"] == "match" if manifest["mode"] == "gpu-regression"
                       else r["status"] == "ok" for r in results))
    value = dict(mode=manifest["mode"], dataset=manifest["dataset"], lifecycle_state=state,
                 quality_status="PASS" if success else "INCOMPLETE" if state != "complete" else "FAIL",
                 selected=len(manifest["selected"]), corpus_count=manifest["corpus_count"],
                 status_counts=dict(counts), queries=results, error=error,
                 baseline=manifest["baseline"], timing_policy=TIMING_POLICY,
                 speedup_reference="historical_saved_cpu_query_seconds", gpu_coverage_gate=False)
    atomic_json(root / "summary.json", value)
    suite = ET.Element("testsuite", name="SQLStorm " + manifest["dataset"],
                       tests=str(len(manifest["selected"])))
    by_id = {r["id"]: r for r in results}
    for qid in manifest["selected"]:
        case = ET.SubElement(suite, "testcase", classname="sqlstorm." + manifest["dataset"], name=qid)
        record = by_id.get(qid, {"status": "not_run"})
        if record["status"] != ("match" if manifest["mode"] == "gpu-regression" else "ok"):
            ET.SubElement(case, "failure", message=record["status"]).text = json.dumps(record)
        ET.SubElement(case, "system-out").text = json.dumps(record)
    if state != "complete":
        case = ET.SubElement(suite, "testcase", name="campaign_completion")
        ET.SubElement(case, "failure", message="incomplete").text = str(error)
    suite.set("tests", str(len(suite)))
    suite.set("failures", str(sum(case.find("failure") is not None for case in suite)))
    ET.ElementTree(suite).write(root / "junit.xml", encoding="utf-8", xml_declaration=True)
    return success


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu-baseline", "gpu-regression"), required=True)
    for name in ("bundle", "data", "spark-home", "python", "output", "baseline"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plugin-jar", type=Path)
    parser.add_argument("--query-timeout", type=int, default=60)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--driver-memory", choices=("4g", "8g"), default="4g")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--capture-mib", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=100000)
    args = parser.parse_args(argv)
    if not (1 <= args.cores <= 4 and 1 <= args.query_timeout <= 120 and 1 <= args.startup_timeout <= 180 and 60 <= args.budget <= 1800
            and 1 <= args.capture_mib <= 32 and 1 <= args.max_rows <= 100000 and args.gpu.isdecimal()):
        parser.error("Regression resource bounds exceeded")
    for name in ("bundle", "data", "spark_home", "python", "output", "baseline"):
        setattr(args, name, getattr(args, name).resolve())
    bundle, _ = load_bundle(args.bundle)
    selected = bundle["selected"]
    if len(selected) > 50:
        parser.error("Initial regression suite is limited to 50 queries")
    if args.mode == "cpu-baseline" and args.baseline.exists():
        parser.error("CPU baselines are immutable; choose a new baseline directory")
    if args.mode == "gpu-regression" and (not args.plugin_jar or not args.plugin_jar.is_file()):
        parser.error("GPU mode needs the candidate plugin JAR")
    if not (args.spark_home / "bin/spark-submit").is_file() or not args.python.is_file():
        parser.error("Spark/Python runtime is unavailable")
    args.tables = TABLES if bundle["dataset"] == "nds" else NDS_H_TABLES
    args.execution_only, args.record_plan = False, False
    identity = data_identity(args.data, args.tables)
    hashes = query_hashes(args.bundle, selected)
    resources = dict(cores=args.cores, driver_memory=args.driver_memory)
    reference, cpu = None, {}
    proof = None
    if args.mode == "cpu-baseline" and bundle.get("adaptations"):
        proof = json.loads((args.bundle / "equivalence.json").read_text())
        validate_equivalence(proof, bundle, identity)
    if args.mode == "gpu-regression":
        reference, cpu = load_reference(args.baseline, bundle, hashes, identity, resources)
        args.expected_spark_version = reference["spark_version"]
        import subprocess
        free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                                        "--format=csv,noheader,nounits"], text=True).splitlines()
        if int(args.gpu) >= len(free) or int(free[int(args.gpu)]) < 15360:
            parser.error("At least 15 GiB free GPU memory is required")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("inputs/source", "logs", "plans", "captures", "scratch"):
        (root / folder).mkdir(parents=True)
    shutil.copytree(args.bundle, root / "inputs/bundle")
    source = Path(__file__).resolve().parent
    for name in ("worker.py", "common.py", "run.py", "regression.py", "adapt.py"):
        shutil.copyfile(source / name, root / "inputs/source" / name)
    baseline_sha = digest(args.baseline / "baseline.json") if reference else None
    if reference:
        (root / "reference").mkdir()
        shutil.copyfile(args.baseline / "baseline.json", root / "reference/baseline.json")
        for record in cpu.values():
            if record["status"] != "ok":
                continue
            target = root / "reference" / record["capture"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.baseline / record["capture"], target)
    manifest = dict(schema_version="sqlstorm.spark.regression.v1", mode=args.mode,
                    dataset=bundle["dataset"], corpus_count=bundle["corpus_count"], selected=selected,
                    upstream_revision=bundle["upstream_revision"], query_sha256=hashes,
                    adaptation=bundle.get("adaptation"), data=identity, resources=resources,
                    adaptation_equivalence=proof if not reference else reference.get("adaptation_equivalence"),
                    baseline=str(args.baseline), baseline_manifest_sha256=baseline_sha,
                    plugin_sha256=digest(args.plugin_jar) if args.mode == "gpu-regression" else None,
                    sql_configuration=SQL_CONF,
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic_json(root / "manifest.json", manifest)
    engine = "cpu" if args.mode == "cpu-baseline" else "gpu"
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    outcomes, ready, error = {}, [], None
    try:
        outcomes, ready, error = execute_engine(args, root, selected, engine, time.monotonic() + args.budget)
    except KeyboardInterrupt:
        error = "Run interrupted; retained partial journals are not a passing reference"
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    complete = set(outcomes) == set(selected) and bool(ready) and error is None
    if (identity != data_identity(args.data, args.tables) or hashes != query_hashes(args.bundle, selected)
            or (reference and baseline_sha != digest(args.baseline / "baseline.json"))
            or (args.mode == "gpu-regression" and manifest["plugin_sha256"] != digest(args.plugin_jar))):
        complete, error = False, "Inputs changed during execution"
    results = []
    if args.mode == "cpu-baseline":
        results = [outcomes[qid] for qid in selected if qid in outcomes]
        if complete and all(r["status"] == "ok" for r in outcomes.values()):
            publish_reference(args.baseline, root, manifest, outcomes, ready)
    else:
        for qid in selected:
            if qid not in outcomes:
                continue
            original = cpu[qid]
            original = dict(original, capture="reference/" + original["capture"]) if original["status"] == "ok" else original
            comparison = compare_results(original, outcomes[qid], root)
            value = dict(id=qid, **comparison, cpu_seconds=cpu[qid].get("query_seconds"),
                         gpu_seconds=outcomes[qid].get("query_seconds"),
                         gpu_status=outcomes[qid]["status"], gpu_error=outcomes[qid].get("error"))
            if comparison["status"] == "match" and value["gpu_seconds"] and value["cpu_seconds"] is not None:
                value["speedup"] = value["cpu_seconds"] / value["gpu_seconds"]
            results.append(value)
    state = "complete" if complete else "incomplete"
    success = make_report(root, manifest, results, state, error)
    print(json.dumps(dict(mode=args.mode, lifecycle_state=state, success=success,
                          selected=len(selected), baseline=str(args.baseline))), flush=True)
    return 2 if not complete else 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
