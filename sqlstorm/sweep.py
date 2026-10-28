# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute a frozen corpus in bounded Spark batches, without a correctness oracle.

Every selected query is attempted, including review-flagged SQL. An explicitly
adapted bundle preserves both original and executable hashes. The sweep itself
does not rewrite SQL or apply compatibility/GPU-plan coverage filters.
Successful execution does not establish CPU/GPU result equality or speedup.
"""
import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

from .bundle import DATASETS
from .common import atomic_json, digest, event
from .run import SQL_CONF, TABLES, dataset_identity, execute_engine

NDS_H_TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp",
                "region", "supplier"]


def query_category(result):
    if result["status"] == "ok":
        return "executed"
    if result["status"] in ("timeout", "crash"):
        return result["status"]
    if result.get("error_type") in ("ParseException", "AnalysisException"):
        return "sql_parse_or_analysis_error"
    return "execution_error_to_triage"


def load_bundle(path, selection=None):
    bundle = json.loads((path / "bundle.json").read_text())
    dataset = bundle["dataset"]
    if dataset not in DATASETS:
        raise ValueError("Unknown dataset")
    corpus = {q["id"]: q for q in bundle["corpus"]}
    if (len(corpus) != DATASETS[dataset][1] or len(corpus) != bundle["corpus_count"]
            or len(corpus) != len(bundle["corpus"])):
        raise ValueError("Incomplete or duplicate corpus metadata")
    selected = bundle["selected"] if selection is None else selection
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Selected query IDs must be unique and nonempty")
    if bundle.get("adaptations"):
        from .adapt import load_bundle as load_adapted
        bundle, corpus = load_adapted(path)
        if set(selected) - set(bundle["selected"]):
            raise ValueError("Selection is not in the frozen adapted bundle")
        return bundle, corpus, selected
    for query_id in selected:
        if query_id not in corpus or not query_id.isdecimal():
            raise ValueError("Unknown query ID: " + query_id)
        query = path / "queries" / (query_id + ".sql")
        if not query.is_file() or digest(query) != corpus[query_id]["sha256"]:
            raise ValueError("Missing/changed original SQL: " + query_id)
    return bundle, corpus, selected


def report(root, manifest, results, state, error=None):
    counts = collections.Counter(r["status"] for r in results.values())
    summary = dict(dataset=manifest["dataset"], lifecycle_state=state,
                   correctness_status="NOT_CHECKED_NO_CPU_ORACLE",
                   corpus_count=manifest["corpus_count"], selected=len(manifest["selected"]),
                   attempted=len(results), remaining=len(manifest["selected"]) - len(results),
                   status_counts=dict(counts),
                   capture_counts=dict(collections.Counter(
                       "complete" if r.get("capture") else r.get("capture_status", "not_collected")
                       for r in results.values())),
                   categories=dict(collections.Counter(query_category(r) for r in results.values())),
                   elapsed_seconds=time.time() - manifest["started_epoch"], error=error)
    atomic_json(root / "summary.json", summary)
    return summary


def prepare_batch(root, source, bundle, selected, number):
    batch = root / "batches" / ("%05d" % number)
    for folder in ("inputs/source", "inputs/bundle/queries", "logs", "plans", "captures", "scratch"):
        (batch / folder).mkdir(parents=True, exist_ok=False)
    for name in ("worker.py", "common.py"):
        shutil.copyfile(source / name, batch / "inputs/source" / name)
    for query_id in selected:
        shutil.copyfile(bundle / "queries" / (query_id + ".sql"),
                        batch / "inputs/bundle/queries" / (query_id + ".sql"))
    return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "data", "spark-home", "plugin-jar", "python", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--engine", choices=("gpu", "cpu"), default="gpu",
                        help="CPU is available for focused failure reproduction only")
    parser.add_argument("--select", help="Optional original IDs for a focused reproduction")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batch-seconds", type=int, default=1200)
    parser.add_argument("--query-timeout", type=int, default=30)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--expected-spark-version", help="Fail startup if the observed Spark version differs")
    parser.add_argument("--collect-results", action="store_true",
                        help="Retain bounded complete typed results when possible; drain a native execution after overflow")
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--capture-mib", type=int, default=32)
    parser.add_argument("--campaign-hours", type=float, default=72)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--driver-memory", choices=("4g", "8g"), default="4g")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args(argv)
    if not (1 <= args.batch_size <= 100 and 1 <= args.query_timeout <= 120
            and 60 <= args.batch_seconds <= 3600 and 0 < args.campaign_hours <= 120
            and 1 <= args.cores <= 4 and args.gpu.isdecimal()
            and 1 <= args.max_rows <= 100000 and 1 <= args.capture_mib <= 32):
        parser.error("Execution/resource bounds exceeded")
    for name in ("bundle", "data", "spark_home", "plugin_jar", "python", "output"):
        setattr(args, name, getattr(args, name).resolve())
    bundle, corpus, selected = load_bundle(args.bundle, args.select.split(",") if args.select else None)
    args.tables = TABLES if bundle["dataset"] == "nds" else NDS_H_TABLES
    args.execution_only = not args.collect_results
    args.full_scan_after_capture_limit = args.collect_results
    args.record_plan = False
    identity = dataset_identity(args.data, 4096 * 1024 ** 2, args.tables)
    if args.engine == "gpu":
        free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                                        "--format=csv,noheader,nounits"], text=True).splitlines()
        if int(args.gpu) >= len(free) or int(free[int(args.gpu)]) < 15360:
            parser.error("At least 15 GiB free GPU memory is required")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    (root / "inputs/source").mkdir(parents=True)
    frozen = root / "inputs/bundle"
    shutil.copytree(args.bundle, frozen)
    source = Path(__file__).resolve().parent
    for name in ("worker.py", "common.py", "run.py", "sweep.py", "bundle.py", "adapt.py"):
        shutil.copyfile(source / name, root / "inputs/source" / name)
    for name in ("LICENSE", "NOTICE", "TPC EULA.txt"):
        if (source.parent / name).is_file():
            shutil.copyfile(source.parent / name, root / "inputs" / name)
    began = time.time()
    manifest = dict(schema_version="sqlstorm.spark.execution_sweep.v1", dataset=bundle["dataset"],
                    upstream_revision=bundle["upstream_revision"], corpus_count=len(corpus),
                    selected=selected, engine=args.engine, started_epoch=began,
                    adaptation=bundle.get("adaptation"),
                    collect_results=args.collect_results,
                    query_sha256={qid: digest(frozen / "queries" / (qid + ".sql")) for qid in selected},
                    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    pid=os.getpid(), host=socket.gethostname(), data=identity,
                    plugin_sha256=digest(args.plugin_jar), sql_configuration=SQL_CONF,
                    limits=dict(batch_size=args.batch_size, batch_seconds=args.batch_seconds,
                                query_seconds=args.query_timeout, cores=args.cores,
                                driver_memory=args.driver_memory),
                    runtime=dict(spark_home=str(args.spark_home), python=str(args.python),
                                 plugin_jar=str(args.plugin_jar), expected_spark_version=args.expected_spark_version),
                    source_sha256={p.name: digest(p) for p in (root / "inputs/source").iterdir()})
    atomic_json(root / "manifest.json", manifest)
    results, pending, number = {}, list(selected), 0
    deadline = time.monotonic() + args.campaign_hours * 3600
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    report(root, manifest, results, "running")
    failure = None
    try:
        while pending and time.monotonic() < deadline:
            number += 1
            current = pending[:args.batch_size]
            batch = prepare_batch(root, root / "inputs/source", frozen, current, number)
            outcomes, ready, error = execute_engine(
                args, batch, current, args.engine, min(deadline, time.monotonic() + args.batch_seconds))
            for query_id, result in outcomes.items():
                if query_id in results:
                    raise ValueError("Query was executed twice: " + query_id)
                result = dict(result, dataset=bundle["dataset"], batch=number,
                              sql_sha256=manifest["query_sha256"][query_id],
                              original_sql_sha256=corpus[query_id]["sha256"],
                              correctness_checked=False, category=query_category(result))
                results[query_id] = result
                event(root / "results.jsonl", result)
            # Spark scratch belongs exclusively to this completed batch. Retain
            # source, SQL, journals and logs, not accumulating shuffle spills.
            shutil.rmtree(batch / "scratch")
            pending = [query_id for query_id in pending if query_id not in results]
            summary = report(root, manifest, results, "running", error)
            print(json.dumps(summary), flush=True)
            if not ready or not outcomes:
                failure = error or "Worker made no progress"
                break
    except KeyboardInterrupt:
        failure = "Campaign interrupted; in-flight batch journals remain available"
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)
    if pending and failure is None:
        failure = "Campaign wall budget exhausted"
    unchanged = identity == dataset_identity(args.data, 4096 * 1024 ** 2, args.tables)
    if not unchanged:
        failure = "Dataset changed during execution"
    if (manifest["plugin_sha256"] != digest(args.plugin_jar) or any(
            digest(frozen / "queries" / (qid + ".sql")) != sha
            for qid, sha in manifest["query_sha256"].items())):
        failure = "Executable SQL/plugin changed during execution"
    state = "complete" if not pending and failure is None else "incomplete"
    summary = report(root, manifest, results, state, failure)
    print(json.dumps(summary), flush=True)
    return 2 if state != "complete" else int(any(r["status"] != "ok" for r in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
