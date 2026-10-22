# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze original SQLStorm queries and the complete corpus denominator."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

try:
    from .common import atomic_json, digest
except ImportError:
    from common import atomic_json, digest

UPSTREAM_REVISION = "b3bb0b96794a6afe9bb8f3ff2b243562b779c40d"
PILOT_IDS = ["16", "30", "37", "47", "109", "111", "131", "164", "192", "218",
             "243", "290", "301", "319", "337", "367", "404", "468", "474", "477"]
REVIEW_PATTERNS = {
    "limit_requires_review": r"\bLIMIT\b",
    "window_requires_review": r"\bOVER\s*\(",
    "nondeterministic_function_review":
        r"\b(random|rand|uuid|first|last|any_value|collect_list|collect_set|array_agg|string_agg|listagg)\s*\(",
}


def review_flags(sql):
    # Conservative screening, not a proof of determinism. False positives stay visible.
    return [name for name, pattern in REVIEW_PATTERNS.items() if re.search(pattern, sql, re.I)]


def build(source, output, selected, revision=UPSTREAM_REVISION):
    source, output = Path(source).resolve(), Path(output).resolve()
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise ValueError("Unexpected SQLStorm source revision: " + actual)
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
                                "--untracked-files=all", "--", "v1.0/tpcds"], text=True):
        raise ValueError("SQLStorm source contains local changes")
    if len(set(selected)) != len(selected) or not selected:
        raise ValueError("Selection must be nonempty and unique")
    manifest = source / "v1.0/tpcds/queries.sql"
    corpus, sql_by_id = [], {}
    for position, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.startswith("\\i "):
            continue
        relative = line.split(maxsplit=1)[1].strip()
        path = (source / relative).resolve()
        if not path.is_relative_to(source / "v1.0/tpcds/queries") or path.suffix != ".sql":
            raise ValueError("Unexpected query path: " + relative)
        query_id = path.stem
        if query_id in sql_by_id:
            raise ValueError("Duplicate original query ID")
        sql = path.read_text()
        sql_by_id[query_id] = sql
        corpus.append(dict(id=query_id, source_path=relative, position=position,
                           sha256=digest(path), review_flags=review_flags(sql)))
    if len(corpus) != 15242:
        raise ValueError("Unexpected original TPC-DS corpus population")
    if set(selected) - sql_by_id.keys():
        raise ValueError("Selection includes unknown query IDs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "queries").mkdir()
    shutil.copyfile(source / "LICENSE", output / "UPSTREAM_LICENSE")
    shutil.copyfile(manifest, output / "corpus-queries.sql")
    for query_id in selected:
        (output / "queries" / (query_id + ".sql")).write_text(sql_by_id[query_id])
    result = dict(schema_version="sqlstorm.spark.bundle.v1", upstream_revision=actual,
                  corpus_manifest_sha256=digest(manifest), corpus_count=len(corpus),
                  corpus=corpus, selected=selected, adaptation="identity_original_sql")
    atomic_json(output / "bundle.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select", default=",".join(PILOT_IDS))
    args = parser.parse_args()
    value = build(args.source, args.output, args.select.split(","))
    print(json.dumps({k: value[k] for k in ("upstream_revision", "corpus_count", "selected")}))


if __name__ == "__main__":
    main()
