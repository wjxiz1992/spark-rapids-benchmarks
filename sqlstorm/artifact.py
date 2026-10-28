# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve Scala 2.13's own Maven snapshot metadata to an immutable JAR URL."""
import argparse
from pathlib import Path
import urllib.parse
import xml.etree.ElementTree as ET


def snapshot_url(base, metadata):
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use a credential-free HTTPS Maven directory")
    artifact, version = parsed.path.rstrip("/").split("/")[-2:]
    root = ET.fromstring(metadata)
    if (artifact != "rapids-4-spark_2.13" or root.findtext("artifactId") != artifact
            or root.findtext("version") != version or not version.endswith("-SNAPSHOT")):
        raise ValueError("Snapshot metadata does not match the requested Scala/version")
    values = {node.findtext("value") for node in root.findall(".//snapshotVersion")
              if node.findtext("extension") == "jar" and node.findtext("classifier") == "cuda12"}
    if len(values) != 1:
        raise ValueError("Exactly one Scala 2.13 CUDA 12 snapshot is required")
    value = values.pop()
    if not value or not value.startswith(version[:-9] + "-") or any(c in value for c in "/\\"):
        raise ValueError("Invalid timestamped snapshot")
    return base.rstrip("/") + "/" + artifact + "-" + value + "-cuda12.jar"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    print(snapshot_url(args.base_url, args.metadata.read_bytes()))
