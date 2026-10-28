# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare captured CPU/GPU discovery sweeps; rewrite semantics remain separate."""
import argparse
import collections
import json
import itertools
from pathlib import Path

from .common import atomic_json, compare_results, digest


def order_review(cpu_capture, gpu_capture, sql_path, sql_hash):
    with cpu_capture.open() as c, gpu_capture.open() as g:
        equal = all(a == b for a, b in itertools.zip_longest(c, g))
    if equal:
        return dict(row_order="identical")
    if digest(sql_path) != sql_hash:
        raise ValueError("Frozen SQL changed before ordering review")
    import sqlglot
    try:
        tree = sqlglot.parse_one(sql_path.read_text(), read="spark")
    except Exception:
        return dict(status="requires_order_review", reason="ordering_contract_not_parsed")
    if tree.args.get("order"):
        # Ties may legally differ, but do not silently treat an observed
        # ordered-output difference as a complete correctness pass.
        return dict(status="requires_order_review", reason="ordered_output_differs")
    return dict(row_order="unspecified_by_query")


def records(root):
    path = root / "results.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines(keepends=True)
              if line.endswith("\n")] if path.exists() else []
    indexed = {r["id"]: r for r in values}
    if len(indexed) != len(values):
        raise ValueError("Duplicate terminal query records")
    return indexed


def compare(cpu, gpu):
    cpu, gpu = Path(cpu), Path(gpu)
    cm, gm = [json.loads((p / "manifest.json").read_text()) for p in (cpu, gpu)]
    if cm["engine"] != "cpu" or gm["engine"] != "gpu":
        raise ValueError("A CPU oracle and a GPU candidate are required")
    for key in ("dataset", "upstream_revision", "selected", "query_sha256", "sql_configuration"):
        if cm[key] != gm[key]:
            raise ValueError("CPU/GPU discovery inputs differ: " + key)
    # Physical mount points may differ between CPU and GPU hosts. Retain every
    # content identity field, including relative paths, sizes and SHA-256 hashes.
    def content(identity):
        return {k: v for k, v in identity.items() if k != "root"} if isinstance(identity, dict) else identity
    if content(cm["data"]) != content(gm["data"]):
        raise ValueError("CPU/GPU discovery inputs differ: data")
    cr, gr = records(cpu), records(gpu)
    values = []
    for qid in cm["selected"]:
        c, g = cr.get(qid), gr.get(qid)
        if c is None or g is None:
            result = dict(status="not_validated", reason="not_executed")
        elif c["status"] != "ok" or g["status"] != "ok":
            result = dict(status="not_validated", reason="execution_error_or_timeout")
        elif not c.get("capture") or not g.get("capture") or c.get("capture_status") == "capture_limit" or g.get("capture_status") == "capture_limit":
            result = dict(status="not_validated", reason="complete_capture_unavailable")
        else:
            cb, gb = [root / "batches" / ("%05d" % record["batch"])
                      for root, record in ((cpu, c), (gpu, g))]
            result = compare_results(c, g, cb, gb)
            if result["status"] == "match":
                result.update(order_review(cb / c["capture"], gb / g["capture"],
                                           cpu / "inputs/bundle/queries" / (qid + ".sql"), cm["query_sha256"][qid]))
        values.append(dict(id=qid, **result, cpu=c, gpu=g))
    return dict(dataset=cm["dataset"], selected=len(cm["selected"]),
                comparison_scope="Spark CPU versus GPU on the identical adapted SQL",
                rewrite_equivalence="NOT_CHECKED_BY_THIS_COMPARISON",
                counts=dict(collections.Counter(v["status"] for v in values)),
                reasons=dict(collections.Counter(v.get("reason", "match") for v in values)), queries=values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.cpu, args.gpu)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "queries"}))
