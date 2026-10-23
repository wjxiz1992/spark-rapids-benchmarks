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
DATASETS = {"nds": ("tpcds", 15242), "nds-h": ("tpch", 17036)}
# Core admitted by the initial 20-query compatibility campaign. The complete
# corpus, including the six failed original queries, remains in every bundle.
PILOT_IDS = ["16", "30", "37", "47", "109", "131", "164", "192", "290", "319",
             "367", "404", "468", "477"]
REVIEW_PATTERNS = {
    "row_selection_requires_review": r"\b(LIMIT|FETCH|OFFSET)\b",
    "window_requires_review": r"\bOVER\s*\(",
    "nondeterministic_function_review":
        r"\b(random|rand|uuid|first|last|any_value|collect_list|collect_set|array_agg|string_agg|listagg)\s*\(",
}


def review_flags(sql):
    # Conservative screening, not a proof of determinism. False positives stay visible.
    return [name for name, pattern in REVIEW_PATTERNS.items() if re.search(pattern, sql, re.I)]


def build(source, output, selected, revision=UPSTREAM_REVISION, dataset="nds"):
    source, output = Path(source).resolve(), Path(output).resolve()
    collection, expected_count = DATASETS[dataset]
    collection_root = source / "v1.0" / collection
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise ValueError("Unexpected SQLStorm source revision: " + actual)
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
                                "--untracked-files=all", "--", "v1.0/" + collection], text=True):
        raise ValueError("SQLStorm source contains local changes")
    if selected is not None and (len(set(selected)) != len(selected) or not selected):
        raise ValueError("Selection must be nonempty and unique")
    manifest = collection_root / "queries.sql"
    corpus, sql_by_id = [], {}
    for position, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.startswith("\\i "):
            continue
        relative = line.split(maxsplit=1)[1].strip()
        path = (source / relative).resolve()
        if not path.is_relative_to(collection_root / "queries") or path.suffix != ".sql":
            raise ValueError("Unexpected query path: " + relative)
        query_id = path.stem
        if query_id in sql_by_id:
            raise ValueError("Duplicate original query ID")
        sql = path.read_text()
        sql_by_id[query_id] = sql
        corpus.append(dict(id=query_id, source_path=relative, position=position,
                           sha256=digest(path), review_flags=review_flags(sql)))
    if len(corpus) != expected_count:
        raise ValueError("Unexpected original " + dataset + " corpus population")
    if selected is None:
        selected = [query["id"] for query in corpus]
    if set(selected) - sql_by_id.keys():
        raise ValueError("Selection includes unknown query IDs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "queries").mkdir()
    shutil.copyfile(source / "LICENSE", output / "UPSTREAM_LICENSE")
    shutil.copyfile(manifest, output / "corpus-queries.sql")
    for query_id in selected:
        (output / "queries" / (query_id + ".sql")).write_text(sql_by_id[query_id])
    result = dict(schema_version="sqlstorm.spark.bundle.v1", dataset=dataset,
                  upstream_collection=collection, upstream_revision=actual,
                  corpus_manifest_sha256=digest(manifest), corpus_count=len(corpus),
                  corpus=corpus, selected=selected, adaptation="identity_original_sql")
    atomic_json(output / "bundle.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="nds")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--select")
    selection.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.dataset == "nds-h" and not args.all and not args.select:
        parser.error("NDS-H requires --all or an explicit --select")
    selected = None if args.all else (args.select or ",".join(PILOT_IDS)).split(",")
    value = build(args.source, args.output, selected, dataset=args.dataset)
    print(json.dumps(dict(upstream_revision=value["upstream_revision"], dataset=value["dataset"],
                          corpus_count=value["corpus_count"], selected_count=len(value["selected"]))))


if __name__ == "__main__":
    main()
