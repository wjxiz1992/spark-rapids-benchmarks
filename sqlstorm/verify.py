# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-time bounded source/adapted comparison using the same DuckDB engine/data.

This validates the rewrite on the chosen dataset. Spark CPU/GPU correctness is
checked separately; this is not a claim of general cross-engine SQL equality.
"""
import argparse
import collections
import datetime
import decimal
import json
from pathlib import Path
import tempfile
import threading

import duckdb

from .adapt import load_bundle
from .common import atomic_json, digest, encoded
from .regression import data_identity
from .run import TABLES
from .sweep import NDS_H_TABLES


def value_record(value):
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["boolean", value]
    if isinstance(value, int):
        return ["integer", str(value)]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, decimal.Decimal):
        return ["decimal", str(value)]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, bytes):
        return ["binary", value.hex()]
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return [type(value).__name__, value.isoformat()]
    if isinstance(value, datetime.timedelta):
        return ["timedelta", value.days, value.seconds, value.microseconds]
    if isinstance(value, (list, tuple)):
        return ["array", [value_record(v) for v in value]]
    if isinstance(value, dict):
        return ["map", sorted([[value_record(k), value_record(v)] for k, v in value.items()], key=encoded)]
    raise TypeError("Unsupported reference result type: " + type(value).__name__)


def consume(connection, sql, seconds, max_rows, max_bytes):
    timer = threading.Timer(seconds, connection.interrupt)
    timer.start()
    try:
        cursor = connection.execute(sql)
        schema = [(column[0], str(column[1])) for column in cursor.description]
        values, count, size = collections.Counter(), 0, 0
        while True:
            batch = cursor.fetchmany(1000)
            if not batch:
                break
            for row in batch:
                record = encoded([value_record(v) for v in row])
                count += 1
                size += len(record.encode("utf-8"))
                if count > max_rows or size > max_bytes:
                    raise ValueError("Complete reference result exceeds the validation budget")
                values[record] += 1
        return schema, values, count
    finally:
        timer.cancel()
        timer.join()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--query-timeout", type=int, default=60)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--capture-mib", type=int, default=16)
    args = parser.parse_args(argv)
    if not (1 <= args.query_timeout <= 120 and 1 <= args.max_rows <= 100000 and 1 <= args.capture_mib <= 32):
        parser.error("Equivalence validation bounds exceeded")
    bundle, _ = load_bundle(args.bundle)
    tables = TABLES if bundle["dataset"] == "nds" else NDS_H_TABLES
    identity = data_identity(args.data, tables)
    results = []
    with tempfile.TemporaryDirectory(prefix="sqlstorm-reference-") as scratch:
        connection = duckdb.connect(config={"threads": 4, "memory_limit": "2GB", "temp_directory": scratch,
                                           "max_temp_directory_size": "512MB"})
        connection.execute("SET TimeZone='UTC'")
        for table in tables:
            files = [str(p) for p in sorted((args.data / table).rglob("*.parquet"))]
            connection.from_parquet(files).create_view(table)
        for query in bundle.get("adaptations", []):
            value = dict(id=query["id"], original_sha256=query["original_sha256"],
                         adapted_sha256=query["adapted_sha256"])
            if query["original_sha256"] == query["adapted_sha256"]:
                value.update(status="identity", result_execution_checked=False)
            else:
                try:
                    before = consume(connection, (args.bundle / "originals" / (query["id"] + ".sql")).read_text(),
                                     args.query_timeout, args.max_rows, args.capture_mib * 1024 ** 2)
                    after = consume(connection, (args.bundle / "queries" / (query["id"] + ".sql")).read_text(),
                                    args.query_timeout, args.max_rows, args.capture_mib * 1024 ** 2)
                    value.update(status="match" if before[:2] == after[:2] else "mismatch",
                                 original_rows=before[2], adapted_rows=after[2],
                                 schema_match=before[0] == after[0], result_execution_checked=True)
                except Exception as error:
                    value.update(status="not_validated", error_type=type(error).__name__, error=str(error)[:4000])
            results.append(value)
            print(json.dumps(value), flush=True)
        connection.close()
    unchanged = identity == data_identity(args.data, tables)
    load_bundle(args.bundle)
    success = unchanged and all(r["status"] in ("identity", "match") for r in results)
    proof = dict(schema_version="sqlstorm.spark.adaptation_equivalence.v1", dataset=bundle["dataset"],
                 engine="DuckDB", engine_version=duckdb.__version__, data=identity,
                 selected=bundle["selected"], queries=results, success=success,
                 scope="Fixed-input source/adapted rewrite comparison; not Spark result validation")
    atomic_json(args.bundle / "equivalence.json", proof)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
