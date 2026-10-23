# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert an existing locally generated TPC-DS raw directory to a new pilot dataset.

Run with CPU spark-submit. Generation is a separate, explicit operation using
the team's existing TPC-DS tools. This does not sample or alter NDS's 3 TB data.
"""
import argparse
from pathlib import Path
import sys

from common import atomic_json, digest
from run import TABLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-version", required=True)
    parser.add_argument("--scale", type=int, choices=(1, 10), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, CharType, VarcharType
    from pyspark.sql.functions import col
    schema_path = Path(__file__).resolve().parent.parent / "nds/nds_schema.py"
    sys.path.insert(0, str(schema_path.parent))
    from nds_schema import get_schemas
    schemas = get_schemas(use_decimal=True)
    spark = SparkSession.builder.appName("SQLStorm CPU input preparation").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    records = []
    try:
        if spark.conf.get("spark.rapids.sql.enabled", "false").lower() == "true":
            raise ValueError("Dataset preparation requires CPU execution")
        for table in TABLES:
            raw = args.raw / (table + ".dat")
            schema = StructType([StructField(f.name,
                StringType() if isinstance(f.dataType, (CharType, VarcharType)) else f.dataType, True)
                for f in schemas[table].fields])
            # dsdgen writes a terminal delimiter. Validate its empty extra field
            # explicitly rather than silently accepting malformed CSV records.
            input_schema = StructType(schema.fields + [StructField("__terminator", StringType())])
            parsed = (spark.read.schema(input_schema).option("delimiter", "|").option("quote", "")
                      .option("encoding", "ISO-8859-1").option("mode", "FAILFAST").csv(str(raw)))
            if parsed.where(col("__terminator").isNotNull()).limit(1).count():
                raise ValueError("Nonempty terminal field in raw table: " + table)
            frame = parsed.drop("__terminator")
            frame.coalesce(1).write.mode("errorifexists").parquet(str(args.output / table))
            with raw.open("rb") as stream:
                expected_rows = sum(1 for _ in stream)
            count = spark.read.parquet(str(args.output / table)).count()
            if count != expected_rows:
                raise ValueError("Converted row count differs: " + table)
            records.append(dict(table=table, rows=count, schema=schema.jsonValue(), raw_sha256=digest(raw)))
        atomic_json(args.output / "dataset.json", dict(schema_version="sqlstorm.spark.dataset.v1",
                    kind="generated_tpcds", scale=args.scale, generator_version=args.generator_version,
                    nds_schema_sha256=digest(schema_path), spark_version=spark.version, tables=records))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
