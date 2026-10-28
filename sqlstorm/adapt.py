# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative SQLStorm syntax adaptations; original bytes stay in the bundle.

Rules produce candidates, not a claim of cross-engine result equivalence.
No rule invents aggregate values, cardinality, ordering, or recursion bounds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import sqlglot
from sqlglot import exp
from sqlglot.dialects import Dialect

from .bundle import DATASETS, UPSTREAM_REVISION
from .common import atomic_json, digest

VERSION = "sqlstorm.spark.adaptation.v1"
PAGE_RULES = [
    (r"\bOFFSET\s+(\d+)\s+ROWS?\s+FETCH\s+(?:FIRST|NEXT)\s+(\d+)\s+ROWS?\s+ONLY\b",
     lambda m: "LIMIT " + m[2] + " OFFSET " + m[1], "literal_offset_fetch"),
    (r"\bFETCH\s+(?:FIRST|NEXT)\s+(\d+)\s+ROWS?\s+ONLY\b",
     lambda m: "LIMIT " + m[1], "literal_fetch"),
    (r"\bOFFSET\s+(\d+)\s+ROWS?\b", lambda m: "OFFSET " + m[1], "literal_offset_rows"),
]


def adapt_sql(sql):
    """Return token-position edits and semantic-review reasons.

    SQLGlot is used to inspect structure; its SQL generator is deliberately not
    used, because transpilation can make additional implicit semantic changes.
    """
    statements = [s for s in sqlglot.parse(sql, read="postgres")
                  if s is not None and not isinstance(s, exp.Semicolon)]
    if len(statements) != 1:
        raise ValueError("Exactly one SELECT query is required")
    tree = statements[0]
    if not isinstance(tree, exp.Query) or tree.find(exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop):
        raise ValueError("Only SELECT queries are accepted")
    tokens = Dialect.get_or_raise("postgres").tokenize(sql)
    starts, ends = {t.start: t for t in tokens}, {t.end + 1: t for t in tokens}
    edits, review = [], set()
    recursive = [w for w in tree.find_all(exp.With) if w.args.get("recursive")]
    removable = True
    for clause in recursive:
        names = [cte.alias.casefold() for cte in clause.expressions]
        for index, cte in enumerate(clause.expressions):
            references = {t.name.casefold() for t in cte.this.find_all(exp.Table) if not t.db}
            if names[index] in references:
                review.add("recursive_cte")
                removable = False
            if references.intersection(names[index + 1:]):
                review.add("forward_cte_reference")
                removable = False
        if clause.args.get("search") or clause.args.get("cycle"):
            review.add("recursive_search_or_cycle")
            removable = False
    recursive_tokens = [t for i, t in enumerate(tokens) if t.token_type.name == "RECURSIVE"
                        and i and tokens[i - 1].token_type.name == "WITH"]
    if recursive and removable:
        if len(recursive_tokens) != len(recursive):
            raise ValueError("Recursive token positions do not match the parsed CTEs")
        edits.extend(dict(start=t.start, end=t.end + 1, replacement="",
                          rule="nonrecursive_keyword") for t in recursive_tokens)
    for pattern, replacement, name in PAGE_RULES:
        for match in re.finditer(pattern, sql, re.I):
            if match.start() not in starts or match.end() not in ends:
                continue
            if starts[match.start()].text.upper() not in ("OFFSET", "FETCH"):
                continue
            if any(match.start() < e["end"] and match.end() > e["start"] for e in edits):
                continue
            edits.append(dict(start=match.start(), end=match.end(),
                              replacement=replacement(match), rule=name))
    stack = []
    for i, token in enumerate(tokens):
        if token.token_type.name == "L_PAREN":
            stack.append(tokens[i - 1].text.upper() if i else "")
        elif token.token_type.name == "R_PAREN":
            if stack:
                stack.pop()
        elif i and i + 1 < len(tokens) and tokens[i - 1].text.upper() == "AS" and stack:
            if stack[-1] not in ("CAST", "TRY_CAST") or tokens[i + 1].token_type.name != "R_PAREN":
                continue
            if token.text.upper() in ("VARCHAR", "TEXT"):
                edits.append(dict(start=token.start, end=token.end + 1, replacement="STRING",
                                  rule="unbounded_" + token.text.lower() + "_cast"))
            elif token.text.upper() in ("CHAR", "CHARACTER"):
                review.add("char_width_semantics")
    for aggregate in tree.find_all(exp.GroupConcat):
        if not isinstance(aggregate.this, exp.Order):
            review.add("unordered_string_aggregate")
    for node in tree.walk():
        if isinstance(node, exp.Anonymous) and node.name.lower() in ("random", "rand", "uuid"):
            review.add("nondeterministic_function")
        if node.__class__.__name__ in ("Rand", "Randn", "Uuid", "CurrentTimestamp", "CurrentDate"):
            review.add("nondeterministic_function")
    if tree.find(exp.Limit, exp.Fetch, exp.Offset):
        review.add("row_selection_ordering")
    if tree.find(exp.Window):
        review.add("window_ordering")
    result = sql
    for edit in sorted(edits, key=lambda e: e["start"], reverse=True):
        result = result[:edit["start"]] + edit["replacement"] + result[edit["end"]:]
    return dict(sql=result, edits=edits, review_reasons=sorted(review),
                source_dialect="postgres_syntax_inspection", adapter_version=VERSION,
                parser_version=sqlglot.__version__)


def load_bundle(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "bundle.json").read_text())
    if manifest["dataset"] not in DATASETS or manifest["upstream_revision"] != UPSTREAM_REVISION:
        raise ValueError("Unexpected query collection/revision")
    corpus = {q["id"]: q for q in manifest["corpus"]}
    if len(corpus) != DATASETS[manifest["dataset"]][1] or len(corpus) != len(manifest["corpus"]):
        raise ValueError("Incomplete/duplicate original corpus")
    if not manifest["selected"] or len(set(manifest["selected"])) != len(manifest["selected"]):
        raise ValueError("Invalid selection")
    adaptations = {q["id"]: q for q in manifest.get("adaptations", [])}
    if adaptations and set(adaptations) != set(manifest["selected"]):
        raise ValueError("Missing adaptation provenance")
    for qid in manifest["selected"]:
        if qid not in corpus or not qid.isdecimal():
            raise ValueError("Unknown/unsafe query ID")
        expected = adaptations[qid]["adapted_sha256"] if adaptations else corpus[qid]["sha256"]
        if digest(path / "queries" / (qid + ".sql")) != expected:
            raise ValueError("Changed executable SQL: " + qid)
        if adaptations and digest(path / "originals" / (qid + ".sql")) != corpus[qid]["sha256"]:
            raise ValueError("Changed original SQL: " + qid)
    return manifest, corpus


def build(bundle, output, selected):
    source, output = Path(bundle).resolve(), Path(output).resolve()
    original, corpus = load_bundle(source)
    if original.get("adaptations"):
        raise ValueError("Adapt from original SQL, not another adapted bundle")
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(original["selected"]):
        raise ValueError("Selection must be unique and available in the source bundle")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("queries", "originals"):
        (output / name).mkdir()
    for name in ("UPSTREAM_LICENSE", "corpus-queries.sql"):
        shutil.copyfile(source / name, output / name)
    changes = []
    for qid in selected:
        raw = (source / "queries" / (qid + ".sql")).read_bytes()
        proposal = adapt_sql(raw.decode("utf-8"))
        adapted = proposal.pop("sql").encode("utf-8")
        (output / "originals" / (qid + ".sql")).write_bytes(raw)
        (output / "queries" / (qid + ".sql")).write_bytes(adapted)
        changes.append(dict(id=qid, source_path=corpus[qid]["source_path"],
                            original_sha256=hashlib.sha256(raw).hexdigest(),
                            adapted_sha256=hashlib.sha256(adapted).hexdigest(), **proposal))
    manifest = dict(original, selected=selected, schema_version="sqlstorm.spark.adapted_bundle.v1",
                    adaptation=VERSION, adaptations=changes)
    atomic_json(output / "bundle.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select", required=True)
    args = parser.parse_args()
    value = build(args.bundle, args.output, args.select.split(","))
    print(json.dumps(dict(dataset=value["dataset"], selected=len(value["selected"]),
                          changed=sum(bool(a["edits"]) for a in value["adaptations"]))))


if __name__ == "__main__":
    main()
