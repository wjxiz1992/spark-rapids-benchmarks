#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import patch

import nds_power


class _DescribeResult:
    def __init__(self, location):
        self.location = location

    def collect(self):
        return [
            {"col_name": "Provider", "data_type": "delta"},
            {"col_name": "Location", "data_type": self.location},
        ]


class _Catalog:
    def __init__(self):
        self.current_database = None
        self.dropped_temp_views = []

    def setCurrentDatabase(self, database):
        self.current_database = database

    def dropTempView(self, table):
        self.dropped_temp_views.append(table)


class _Conf:
    def get(self, key):
        if key != "spark.app.id":
            raise AssertionError("Unexpected key: {}".format(key))
        return "test-app"


class _SparkSession:
    def __init__(self, registered_location):
        self.catalog = _Catalog()
        self.conf = _Conf()
        self.registered_location = registered_location
        self.statements = []

    def sql(self, statement):
        self.statements.append(statement)
        if statement.startswith("DESCRIBE TABLE EXTENDED "):
            return _DescribeResult(self.registered_location)
        return None


class DeltaTableRegistrationTest(unittest.TestCase):
    def test_database_name_uses_normalized_warehouse_location(self):
        first = "abfss://nds@account.dfs.core.windows.net/delta_sf1/"
        equivalent = "ABFSS://NDS@ACCOUNT.DFS.CORE.WINDOWS.NET//delta_sf1//"
        different = "abfss://nds@account.dfs.core.windows.net/delta_sf3k"

        self.assertEqual(
            nds_power._delta_database_name(first),
            nds_power._delta_database_name(equivalent))
        self.assertNotEqual(
            nds_power._delta_database_name(first),
            nds_power._delta_database_name(different))

    @patch("nds_power._qualify_delta_location", return_value="file:///tmp/delta_sf3k")
    @patch("nds_power.get_schemas", return_value={"customer": None})
    def test_registers_in_warehouse_database_and_checks_location(self, _, __):
        warehouse = "/tmp/delta_sf3k/"
        table_path = warehouse.rstrip("/") + "/customer"
        spark = _SparkSession("file:///tmp/delta_sf3k/customer")

        timings = nds_power.register_delta_tables(spark, warehouse, [])

        database = nds_power._delta_database_name("file:///tmp/delta_sf3k")
        self.assertEqual(database, spark.catalog.current_database)
        self.assertEqual(["customer"], spark.catalog.dropped_temp_views)
        self.assertEqual(
            "CREATE DATABASE IF NOT EXISTS `{}`".format(database),
            spark.statements[0])
        self.assertEqual(
            "CREATE TABLE IF NOT EXISTS `{}`.`customer` USING DELTA LOCATION '{}'".format(
                database, table_path),
            spark.statements[1])
        self.assertEqual(
            "DESCRIBE TABLE EXTENDED `{}`.`customer`".format(database),
            spark.statements[2])
        self.assertEqual("Register customer", timings[0][1])

    @patch("nds_power._qualify_delta_location",
           return_value="abfss://nds@account.dfs.core.windows.net/delta_sf3k")
    @patch("nds_power.get_schemas", return_value={"customer": None})
    def test_rejects_existing_table_at_another_location(self, _, __):
        warehouse = "abfss://nds@account.dfs.core.windows.net/delta_sf3k"
        spark = _SparkSession(
            "abfss://nds@account.dfs.core.windows.net/delta_sf1/customer")

        with self.assertRaisesRegex(RuntimeError, "Refusing to run maintenance"):
            nds_power.register_delta_tables(spark, warehouse, [])


if __name__ == "__main__":
    unittest.main()
