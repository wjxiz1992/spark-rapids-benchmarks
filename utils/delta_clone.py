# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-scoped managed shallow clones for immutable Delta benchmark inputs."""

import uuid


def _quote_identifier(identifier):
    return "`{}`".format(identifier.replace("`", "``"))


class DeltaCloneScope:
    """Own a unique database of managed shallow clones for one benchmark run."""

    def __init__(self, spark_session, database_prefix):
        self._spark_session = spark_session
        self.previous_database = spark_session.catalog.currentDatabase()
        self.database_name = "{}_{}".format(database_prefix, uuid.uuid4().hex)
        self._database_identifier = _quote_identifier(self.database_name)
        self._is_open = False

        create_database_sql = "CREATE DATABASE {}".format(self._database_identifier)
        print(create_database_sql)
        spark_session.sql(create_database_sql)
        self._is_open = True
        try:
            spark_session.catalog.setCurrentDatabase(self.database_name)
        except Exception:
            self.close()
            raise

    def clone_table(self, source_path, table_name):
        """Create a managed shallow clone whose unqualified name resolves in this scope."""
        table_identifier = "{}.{}".format(
            self._database_identifier, _quote_identifier(table_name))
        source_identifier = "delta.{}".format(_quote_identifier(source_path))
        clone_sql = "CREATE TABLE {} SHALLOW CLONE {}".format(
            table_identifier, source_identifier)
        print(clone_sql)
        self._spark_session.catalog.dropTempView(table_name)
        self._spark_session.sql(clone_sql)

    def close(self):
        """Restore the caller's database and delete all managed clone tables."""
        if not self._is_open:
            return
        self._spark_session.catalog.setCurrentDatabase(self.previous_database)
        drop_database_sql = "DROP DATABASE {} CASCADE".format(self._database_identifier)
        print(drop_database_sql)
        self._spark_session.sql(drop_database_sql)
        self._is_open = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
