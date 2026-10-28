# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from sqlstorm.artifact import snapshot_url


class Artifacts(unittest.TestCase):
    base = "https://example.test/com/nvidia/rapids-4-spark_2.13/26.12.0-SNAPSHOT"
    xml = """<metadata><artifactId>rapids-4-spark_2.13</artifactId><version>26.12.0-SNAPSHOT</version>
      <versioning><snapshotVersions><snapshotVersion><classifier>cuda12</classifier><extension>jar</extension>
      <value>26.12.0-20261027.183146-15</value></snapshotVersion><snapshotVersion><classifier>sources</classifier>
      <extension>jar</extension><value>26.12.0-20261027.183146-99</value></snapshotVersion></snapshotVersions></versioning></metadata>"""

    def test_exact_cuda_jar_uses_its_own_timestamp(self):
        self.assertTrue(snapshot_url(self.base, self.xml).endswith("rapids-4-spark_2.13-26.12.0-20261027.183146-15-cuda12.jar"))

    def test_other_scala_metadata_or_missing_cuda_snapshot_is_rejected(self):
        for xml in (self.xml.replace("2.13", "2.12"), self.xml.replace("cuda12", "cuda13")):
            with self.assertRaises(ValueError):
                snapshot_url(self.base, xml)


if __name__ == "__main__":
    unittest.main()
