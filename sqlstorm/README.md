# SQLStorm Spark correctness pilot

This opt-in pilot compares original SQLStorm TPC-DS queries on CPU Spark and
cuDF for Apache Spark. It is a bounded compatibility/correctness diagnostic,
not a TPC score or controlled performance benchmark.

## Scope

- Upstream SQLStorm is pinned to `b3bb0b96794a6afe9bb8f3ff2b243562b779c40d`.
  The bundle retains the full 15,242-query denominator and source hashes.
- The initial core contains 14 unchanged queries admitted by the first 20-query
  compatibility campaign. Five original queries (111, 218, 243, 301, 474) use
  FETCH/OFFSET syntax rejected by Spark 3.5.7 and require row-selection review;
  query 337 failed Spark analysis and contains a correlated scalar subquery
  requiring further compatibility triage. Their original
  failures remain in [the discovery campaign](https://prod.blsm.nvidia.com/sw-gpu-spark-jenkins/job/allxu-sqlstorm-tpcds-pilot-spark2a/2/).
  These cases remain in every full-corpus denominator and are not passing tests.
  Conservative screening rejects selected LIMIT/FETCH/OFFSET, window, and known
  nondeterministic aggregate/function shapes.
  Screening is not a general determinism proof. Unselected/review-needed queries
  remain visible; the pilot does not claim full-corpus validation.
- A worker reuses one local Spark application per engine batch. CPU and GPU
  use separate processes, warehouses and scratch paths. Hybrid execution is
  allowed. Nonempty matches require GPU operators with positive row/batch metrics.
  Empty matches remain visible as value checks and do not count as GPU coverage.
  A campaign with only empty matches cannot pass the pilot.
- Only small **local** Parquet datasets and `local[1..4]` are supported. This
  pilot rejects datasets exceeding its configured byte budget (2 GiB by default,
  at most 4 GiB), more than 50 selected queries and wall budgets over 30 minutes.
  It does not implement distributed 3 TB execution or recurring scheduling.
- The JVM heap is 4 GiB and the RAPIDS GPU pool is capped at 30%. These are
  engine limits, not an OS/cgroup total-memory limit. A shared-cluster run must
  hold `SPARK2A_CLUSTER`; resource settings alone do not isolate disk or caches.

## Prepare a frozen bundle

Clone SQLStorm and check out the pinned commit above. The upstream TPC-DS
directory must be clean. The source SQL and upstream MIT license are retained.

```bash
python3 -m sqlstorm.bundle --source /path/to/SQLStorm --output /new/bundle \
  --select 16,30,37,47,109
```

## Prepare data or use existing small NDS Parquet data

`prepare.py` reads existing raw `.dat` files produced by the team's separately
installed TPC-DS tools. It does not acquire tools, generate data automatically,
sample the 3 TB dataset, or replace existing outputs. Follow the existing NDS
TPC licensing and data-generation instructions. Use scale 1 or 10.

```bash
spark-submit --master 'local[4]' --driver-memory 4g \
  --conf spark.sql.csv.parser.columnPruning.enabled=false \
  sqlstorm/prepare.py --raw /generated/sf1 --output /new/parquet-sf1 \
  --scale 1 --generator-version ACTUAL_GENERATOR_VERSION
```

Conversion uses NDS Decimal schemas, maps Char/Varchar fields to String, handles
the terminal pipe as a checked extra field, and verifies table row counts.
`dataset.json` records raw-input hashes, generator label and NDS schema hash.
For preexisting data without generator provenance, use an honest descriptive
label. The runtime records actual file hashes and observed schemas in either case.

## Run

Use Python 3.10+ supported by the selected Spark version. The selected
GPU must have at least 15 GiB free VRAM. Use a new output directory for each run.

```bash
python3 -m sqlstorm.run --bundle /new/bundle --data /new/parquet-sf1 \
  --dataset-label generated-tpcds-sf1-decimal \
  --spark-home /path/to/spark --plugin-jar /path/to/plugin.jar \
  --python /path/to/python3.10 --output /new/campaign \
  --whole-run-timeout 1200 --query-timeout 60
```

The plugin JAR must contain the matching JNI/cuDF runtime or be packaged as the
usual self-contained distribution JAR. Existing Spark default configuration
files are hashed. For isolation set `SPARK_CONF_DIR` to a dedicated empty folder;
the runner explicitly sets its SQL and resource settings.

`manifest.json`, CPU/GPU fsynced journals, complete/partial captures, executed
plans with SQL metrics, original SQL, frozen worker source, logs, `summary.json`
and `junit.xml` stay together. Inputs and the plugin are rehashed after execution.
A query that starts without a terminal record is recorded as a timeout/crash
after terminating its worker process group. Startup failure and remaining
unscheduled engine attempts remain incomplete. There is no automatic retry of
an already terminal query and no live resume.

## Interpretation

`spark_typed_unordered_multiset_v1` compares original logical schemas and all
captured rows. Row order is ignored; duplicate multiplicity and nested values
are preserved. Integers and Decimal values remain exact. All floating NaNs
compare equal, infinities retain signs, and signed floating zero is distinct.
No tolerance is applied. Missing, truncated, changed or unsupported captures
cannot pass. A mismatch is a finding to investigate, not independent proof of
a cuDF defect. CPU query errors and unsupported SQL remain explicit.

Plan evidence is captured after result consumption, including AQE/query stages
and their runtime metrics. It reports meaningful GPU operators separately from
results; it is not a CUDA kernel trace. Empty matches are explicitly marked.

Exit codes: 0 means all selected queries matched, every nonempty result has GPU
output-metric evidence, and at least one query had nonempty GPU work; 1 means
completed selected work has execution/value/coverage failures; 2 means incomplete
or invalid evidence. Interrupted runs preserve their incomplete report.

`ci.py` stages the same code on Spark2a worker u13 using the existing Jenkins
SSH credential **only on the agent**. The pipeline holds `SPARK2A_CLUSTER`,
resolves one plugin JAR, records its SHA256, enforces an outer timeout, and
archives evidence. It does not redeploy or restart the NDS cluster, touch the
3 TB data, or send notifications. Remote evidence is retained at the printed
unique path for investigation.

## Validation

```bash
python3 -m unittest discover -s sqlstorm/tests -v
```

Tests include exact-value/multiplicity counterexamples, precision boundaries,
capture corruption, nested nulls, real child-process crash/deadline handling,
data-size rejection and remote-shell parameter quoting. A real Spark CPU/GPU
pilot is also required before publishing runner changes.
