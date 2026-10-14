# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check run-scoped Delta clone registration in both power drivers."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch

import nds_maintenance
import nds_power


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nds-h'))
spec = importlib.util.spec_from_file_location('nds_h_delta_power', ROOT / 'nds-h/nds_h_power.py')
nds_h_power = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nds_h_power)


class DeltaCloneRegistrationTest(unittest.TestCase):
    def test_bundled_delta_templates_use_clone_capable_release(self):
        coordinate = 'io.delta:delta-core_2.12:2.3.0'
        template_names = (
            'convert_submit_cpu_delta.template',
            'maintenance_delta.template',
            'power_run_cpu_delta.template',
            'power_run_gpu_delta.template',
        )

        for template_name in template_names:
            with self.subTest(template=template_name):
                template = (ROOT / 'nds' / template_name).read_text()
                self.assertIn(coordinate, template)
                self.assertNotIn('io.delta:delta-core_2.12:2.1.1', template)

    def test_nds_registers_every_delta_path_in_one_clone_scope(self):
        spark = Mock()
        spark.conf.get.return_value = 'application-nds'
        scope = Mock()
        with patch.object(nds_power, 'DeltaCloneScope', return_value=scope) as scope_type, \
                patch.object(nds_power, 'get_schemas', return_value={'customer': None,
                                                                    'store_sales': None}), \
                patch.object(nds_power.time, 'time', side_effect=[1.000, 1.010, 2.000, 2.025]):
            times, returned_scope = nds_power.register_delta_tables(
                spark, 'abfss://container/reference/', [])

        scope_type.assert_called_once_with(spark, 'nds_delta_run')
        self.assertEqual(scope.clone_table.call_args_list, [
            call('abfss://container/reference/customer', 'customer'),
            call('abfss://container/reference/store_sales', 'store_sales'),
        ])
        self.assertEqual(times, [
            ('application-nds', 'ShallowClone customer', 10),
            ('application-nds', 'ShallowClone store_sales', 25),
        ])
        self.assertIs(returned_scope, scope)

    def test_nds_h_closes_clone_scope_if_registration_fails(self):
        spark = Mock()
        spark.conf.get.return_value = 'application-nds-h'
        scope = Mock()
        scope.clone_table.side_effect = RuntimeError('clone failed')
        with patch.object(nds_h_power, 'DeltaCloneScope', return_value=scope), \
                patch.object(nds_h_power, 'get_schemas', return_value={'lineitem': None}):
            with self.assertRaisesRegex(RuntimeError, 'clone failed'):
                nds_h_power.register_delta_tables(spark, '/reference', [])

        scope.clone_table.assert_called_once_with('/reference/lineitem', 'lineitem')
        scope.close.assert_called_once_with()

    def test_nds_power_closes_clone_scope_on_failure_and_stops_session(self):
        spark = Mock()
        spark.conf.get.return_value = 'application-nds'
        builder = Mock()
        builder.appName.return_value.getOrCreate.return_value = spark
        scope = Mock()
        error = RuntimeError('query setup failed')

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(nds_power.SparkSession, 'builder', builder), \
                patch.object(nds_power, 'register_delta_tables', return_value=([], scope)), \
                patch.object(nds_power, 'check_json_summary_folder', side_effect=error):
            with self.assertRaises(RuntimeError) as stopped:
                nds_power.run_query_stream(
                    '/reference', None, {}, str(Path(directory) / 'times.csv'), None,
                    None, None, 0, 1, ['logical'], input_format='delta',
                    delta_unmanaged=True)

        self.assertIs(stopped.exception, error)
        scope.close.assert_called_once_with()
        spark.stop.assert_called_once_with()

    def test_nds_h_power_closes_clone_scope_on_failure_and_stops_session(self):
        spark = Mock()
        spark.conf.get.return_value = 'application-nds-h'
        builder = Mock()
        builder.appName.return_value.getOrCreate.return_value = spark
        scope = Mock()
        error = RuntimeError('query setup failed')

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(nds_h_power.SparkSession, 'builder', builder), \
                patch.object(nds_h_power, 'register_delta_tables', return_value=([], scope)), \
                patch.object(nds_h_power, 'check_json_summary_folder', side_effect=error):
            with self.assertRaises(RuntimeError) as stopped:
                nds_h_power.run_query_stream(
                    '/reference', None, {}, str(Path(directory) / 'times.csv'),
                    None, 0, 1, ['logical'], 'delta')

        self.assertIs(stopped.exception, error)
        scope.close.assert_called_once_with()
        spark.stop.assert_called_once_with()

    def test_maintenance_closes_clone_scope_on_query_failure_and_stops_context(self):
        spark = Mock()
        spark.sparkContext.applicationId = 'application-maintenance'
        scope = Mock()
        report = Mock()
        error = RuntimeError('maintenance query failed')
        report.report_on.side_effect = error

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(nds_maintenance, 'register_delta_tables',
                             return_value=([], scope)), \
                patch.object(nds_maintenance, 'PysparkBenchReport', return_value=report), \
                patch.object(nds_maintenance, 'setQueryName'):
            with self.assertRaises(RuntimeError) as stopped:
                nds_maintenance.run_query(
                    spark, {'DF_CS': ['DELETE FROM catalog_sales']},
                    str(Path(directory) / 'times.csv'), None, None,
                    '/reference', 'delta', False, delta_unmanaged=True)

        self.assertIs(stopped.exception, error)
        scope.close.assert_called_once_with()
        spark.sparkContext.stop.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
