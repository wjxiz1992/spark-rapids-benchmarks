# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One isolated local Spark application per CPU/GPU batch; supervised by run.py."""
import argparse
import datetime
import json
import os
from pathlib import Path
import time

from common import atomic_json, digest, encoded, error_record, event, scalar


def plan_evidence(df):
    plan = df._jdf.queryExecution().executedPlan()
    text = plan.toString()
    nodes, seen = [], set()

    def visit(node):
        key = (node.getClass().getName(), node.id())
        if key in seen:
            return
        seen.add(key)
        values = {}
        iterator = node.metrics().iterator()
        while iterator.hasNext():
            pair = iterator.next()
            values[pair._1()] = pair._2().value()
        nodes.append(dict(class_name=key[0], name=node.nodeName(), metrics=values))
        if key[0].endswith("AdaptiveSparkPlanExec"):
            visit(node.executedPlan())
        elif key[0].endswith("QueryStageExec"):
            visit(node.plan())
        children = node.children().iterator()
        while children.hasNext():
            visit(children.next())

    visit(plan)
    wrappers = {"GpuColumnarToRowExec", "GpuRowToColumnarExec", "GpuBringBackToHostExec"}
    gpu = sorted({n["class_name"].split(".")[-1] for n in nodes
                  if n["class_name"].split(".")[-1].startswith("Gpu")
                  and n["class_name"].split(".")[-1] not in wrappers})
    active = sorted({n["class_name"].split(".")[-1] for n in nodes
                     if n["class_name"].split(".")[-1] in gpu
                     and any(n["metrics"].get(k, 0) > 0 for k in
                             ("numOutputRows", "numInputRows", "numOutputBatches", "numInputBatches"))})
    return dict(text=text, nodes=nodes, gpu_operators=gpu, gpu_active_operators=active,
                evidence="executed_spark_plan_after_result_consumption",
                caveat="Plan evidence is not a CUDA kernel trace; hybrid CPU/GPU execution is allowed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    root = Path(job["output"])
    journal = root / (job["engine"] + ".jsonl")
    os.environ["TZ"] = "UTC"
    if hasattr(time, "tzset"):
        time.tzset()
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("SQLStorm pilot " + job["engine"]).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    enabled = spark.conf.get("spark.rapids.sql.enabled", "false").lower() == "true"
    if enabled != (job["engine"] == "gpu"):
        raise RuntimeError("Observed RAPIDS setting differs from requested engine")
    tables = []
    for table in job["tables"]:
        frame = spark.read.parquet(str(Path(job["data"]) / table))
        frame.createOrReplaceTempView(table)
        tables.append(dict(name=table, schema=frame.schema.jsonValue()))
    event(journal, dict(type="ready", engine=job["engine"], spark_version=spark.version,
                        application_id=spark.sparkContext.applicationId, tables=tables,
                        configuration={key: spark.conf.get(key) for key in job["sql_configuration"]}))
    try:
        for query in job["queries"]:
            attempt = query["attempt_id"]
            started = time.monotonic()
            event(journal, dict(type="started", engine=job["engine"], id=query["id"],
                                attempt_id=attempt, utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            capture = root / "captures" / (job["engine"] + "-" + attempt + ".jsonl")
            result = dict(type="finished", engine=job["engine"], id=query["id"],
                          attempt_id=attempt, status="error", row_count=0)
            spark.sparkContext.setJobGroup(attempt, "SQLStorm original " + query["id"], interruptOnCancel=True)
            try:
                df = spark.sql((root / "inputs/bundle/queries" / (query["id"] + ".sql")).read_text())
                schema = df.schema.jsonValue()
                count, size = 0, 0
                with capture.open("x", encoding="utf-8") as out:
                    for row in df.toLocalIterator(prefetchPartitions=False):
                        value = encoded([scalar(row[i], field["type"])
                                         for i, field in enumerate(schema["fields"])]) + "\n"
                        size += len(value.encode("utf-8"))
                        if count >= job["max_rows"] or size > job["capture_bytes"]:
                            raise CaptureLimit("Complete result exceeded capture budget")
                        out.write(value)
                        count += 1
                    out.flush()
                    os.fsync(out.fileno())
                evidence = plan_evidence(df)
                plan_file = root / "plans" / (job["engine"] + "-" + attempt + ".json")
                atomic_json(plan_file, evidence)
                result.update(status="ok", schema=schema, row_count=count,
                              capture=str(capture.relative_to(root)), capture_bytes=size,
                              capture_sha256=digest(capture), plan=str(plan_file.relative_to(root)),
                              gpu_operators=evidence["gpu_operators"],
                              gpu_active_operators=evidence["gpu_active_operators"])
            except CaptureLimit as exc:
                spark.sparkContext.cancelJobGroup(attempt)
                result.update(status="capture_limit", error=str(exc))
            except Exception as exc:
                result.update(error_record(exc))
            result["elapsed_seconds"] = time.monotonic() - started
            if result["elapsed_seconds"] > job["query_timeout"]:
                result.update(status="timeout", error="Query exceeded deadline including capture")
            event(journal, result)
            spark.sparkContext.setLocalProperty("spark.jobGroup.id", None)
    finally:
        spark.stop()


class CaptureLimit(RuntimeError):
    pass


if __name__ == "__main__":
    main()
