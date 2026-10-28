# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from sqlstorm.adapt import adapt_sql


class Adaptation(unittest.TestCase):
    def test_only_token_positions_are_changed(self):
        sql = "-- FETCH FIRST 9 ROWS ONLY\r\nSELECT 'CAST(x AS TEXT)', x FROM t OFFSET 2 ROWS FETCH NEXT 3 ROWS ONLY;\r\n"
        result = adapt_sql(sql)
        self.assertEqual("-- FETCH FIRST 9 ROWS ONLY\r\nSELECT 'CAST(x AS TEXT)', x FROM t LIMIT 3 OFFSET 2;\r\n", result["sql"])

    def test_nonrecursive_keyword_does_not_change_other_bytes(self):
        self.assertEqual("WITH  t AS (SELECT 1) SELECT * FROM t",
                         adapt_sql("WITH RECURSIVE t AS (SELECT 1) SELECT * FROM t")["sql"])

    def test_comment_after_final_semicolon_is_preserved(self):
        sql = "SELECT CAST(x AS TEXT) FROM t; -- original comment"
        self.assertEqual("SELECT CAST(x AS STRING) FROM t; -- original comment", adapt_sql(sql)["sql"])

    def test_recursion_and_forward_dependencies_are_not_erased(self):
        sql = "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM t WHERE n<3) SELECT * FROM t"
        self.assertEqual(sql, adapt_sql(sql)["sql"])
        self.assertIn("recursive_cte", adapt_sql(sql)["review_reasons"])
        sql = "WITH RECURSIVE a AS (SELECT * FROM b), b AS (SELECT 1) SELECT * FROM a"
        self.assertEqual(sql, adapt_sql(sql)["sql"])
        self.assertIn("forward_cte_reference", adapt_sql(sql)["review_reasons"])

    def test_character_width_and_sized_casts_are_preserved(self):
        sql = "SELECT CAST(x AS VARCHAR), CAST(y AS TEXT), CAST(z AS CHAR), CAST(x AS VARCHAR(2)) FROM t"
        self.assertEqual("SELECT CAST(x AS STRING), CAST(y AS STRING), CAST(z AS CHAR), CAST(x AS VARCHAR(2)) FROM t",
                         adapt_sql(sql)["sql"])
        self.assertIn("char_width_semantics", adapt_sql(sql)["review_reasons"])

    def test_no_sort_or_first_is_invented(self):
        sql = "SELECT STRING_AGG(x, ',') FROM t"
        result = adapt_sql(sql)
        self.assertEqual(sql, result["sql"])
        self.assertIn("unordered_string_aggregate", result["review_reasons"])

    def test_ordered_string_aggregate_does_not_require_unordered_review(self):
        result = adapt_sql("SELECT STRING_AGG(x, ',' ORDER BY x) FROM t")
        self.assertNotIn("unordered_string_aggregate", result["review_reasons"])

    def test_nondeterministic_queries_remain_visible(self):
        result = adapt_sql("SELECT random() FROM t LIMIT 1")
        self.assertIn("nondeterministic_function", result["review_reasons"])
        self.assertIn("row_selection_ordering", result["review_reasons"])

    def test_statements_with_side_effects_are_rejected(self):
        for sql in ("DELETE FROM t", "SELECT 1; DROP TABLE t", "WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x"):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                adapt_sql(sql)


if __name__ == "__main__":
    unittest.main()
