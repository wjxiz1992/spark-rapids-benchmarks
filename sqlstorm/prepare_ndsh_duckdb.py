# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate independent SF1 data with DuckDB TPCH, cast to the NDS-H schema.

Requires duckdb and pyspark for Python schema types, not a running Spark JVM.
Existing datasets are never replaced. This is not an official TPC result.
"""
import argparse
import importlib.util
from pathlib import Path
import shutil

from .common import atomic_json, digest

COUNTS = dict(region=5, nation=25, supplier=10000, customer=150000, part=200000,
              partsupp=800000, orders=1500000, lineitem=6001215)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import duckdb

    schema_file = Path(__file__).resolve().parent.parent / "nds-h/nds_h_schema.py"
    spec = importlib.util.spec_from_file_location("nds_h_schema", schema_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schemas = module.get_schemas()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    db = duckdb.connect()
    db.execute("SET threads=4")
    db.execute("SET memory_limit='4GB'")
    db.execute("LOAD tpch")
    db.execute("CALL dbgen(sf=1)")
    tables = []
    for table, expected in COUNTS.items():
        actual = db.execute('SELECT count(*) FROM "' + table + '"').fetchone()[0]
        if actual != expected:
            raise ValueError("Unexpected SF1 row count: " + table)
        fields = [field for field in schemas[table].fields if field.name != "ignore"]
        types = {"string": "VARCHAR", "int": "INTEGER", "bigint": "BIGINT", "date": "DATE"}
        projection = ",".join('CAST("' + f.name + '" AS '
                              + types.get(f.dataType.simpleString(), f.dataType.simpleString())
                              + ') AS "' + f.name + '"' for f in fields)
        folder = root / table
        folder.mkdir()
        output = folder / "part-00000.parquet"
        quoted = str(output).replace("'", "''")
        db.execute('COPY (SELECT ' + projection + ' FROM "' + table + '") TO '
                   + "'" + quoted + "' (FORMAT PARQUET, COMPRESSION SNAPPY)")
        stored = db.execute("SELECT count(*) FROM read_parquet(?)", [str(output)]).fetchone()[0]
        if stored != expected:
            raise ValueError("Parquet row count differs: " + table)
        tables.append(dict(table=table, rows=stored, path=str(output.relative_to(root)),
                           sha256=digest(output), schema=schemas[table].jsonValue()))
        print(table, stored, flush=True)
    db.close()
    shutil.copyfile(schema_file.parent.parent / "TPC EULA.txt", root / "TPC EULA.txt")
    atomic_json(root / "dataset.json", dict(dataset="nds-h", scale=1,
                generator="DuckDB tpch extension dbgen", generator_version=duckdb.__version__,
                schema_sha256=digest(schema_file), tables=tables,
                disclaimer="Not comparable to official TPC Benchmark Results"))


if __name__ == "__main__":
    main()
