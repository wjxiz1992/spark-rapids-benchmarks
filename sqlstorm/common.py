# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned, exact result records for the bounded Spark SQLStorm pilot."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

POLICY = "spark_typed_unordered_multiset_v1"


def error_record(error):
    # PySpark captured exceptions can have an empty repr(). Preserve the actual
    # message separately so a long analyzed plan cannot displace the root cause.
    return dict(error_type=type(error).__name__, error=str(error),
                traceback=traceback.format_exc()[-12000:])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, allow_nan=False, indent=2)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def event(path, value):
    with Path(path).open("a", encoding="utf-8") as out:
        out.write(encoded(value) + "\n")
        out.flush()
        os.fsync(out.fileno())


def scalar(value, dtype):
    """Preserve values without converting decimals or integers through floats.

    dtype is Spark DataType.jsonValue(). Map order is ignored; list order,
    duplicate map entries in the representation, and signed floating zero remain.
    Spark timestamps have microsecond precision. Workers run with TZ=UTC.
    """
    if value is None:
        return ["null"]
    if isinstance(dtype, dict):
        kind = dtype["type"]
        if kind == "struct":
            return [kind, [scalar(value[i], f["type"]) for i, f in enumerate(dtype["fields"])]]
        if kind == "array":
            return [kind, [scalar(v, dtype["elementType"]) for v in value]]
        if kind == "map":
            pairs = [[scalar(k, dtype["keyType"]), scalar(v, dtype["valueType"])]
                     for k, v in value.items()]
            return [kind, sorted(pairs, key=encoded)]
        raise TypeError("Unsupported result type: " + str(dtype))
    if dtype.startswith("decimal("):
        return [dtype, str(value)]
    if dtype in ("float", "double"):
        return [dtype, "nan" if math.isnan(value) else value.hex()]
    if dtype in ("byte", "short", "integer", "long", "boolean", "string"):
        return [dtype, value]
    if dtype in ("timestamp", "timestamp_ntz", "date"):
        return [dtype, value.isoformat()]
    if dtype == "binary":
        return [dtype, bytes(value).hex()]
    raise TypeError("Unsupported result type: " + str(dtype))


def compare_results(cpu, gpu, folder, gpu_folder=None):
    """Compare verified complete captures, never just their hashes or row counts."""
    base = {"policy": POLICY, "cpu_attempt": cpu.get("attempt_id"),
            "gpu_attempt": gpu.get("attempt_id")}
    if cpu.get("status") != "ok" or gpu.get("status") != "ok":
        return dict(base, status="not_validated", reason="execution_or_capture_incomplete")
    captures = []
    for item, base_folder in zip((cpu, gpu), (folder, gpu_folder if gpu_folder is not None else folder)):
        path = (Path(base_folder) / item["capture"]).resolve()
        if not path.is_relative_to(Path(base_folder).resolve()):
            raise ValueError("Capture escaped campaign")
        if not path.is_file() or digest(path) != item["capture_sha256"]:
            return dict(base, status="not_validated", reason="capture_integrity")
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if len(rows) != item["row_count"]:
            return dict(base, status="not_validated", reason="capture_row_count")
        captures.append(collections.Counter(encoded(row) for row in rows))
    if cpu["schema"] != gpu["schema"]:
        return dict(base, status="mismatch", reason="schema")
    if captures[0] != captures[1]:
        missing = captures[0] - captures[1]
        extra = captures[1] - captures[0]
        return dict(base, status="mismatch", reason="values_or_multiplicity",
                    cpu_only=list(missing.items())[:3], gpu_only=list(extra.items())[:3])
    return dict(base, status="match", row_count=cpu["row_count"], empty=cpu["row_count"] == 0)
