# SQLStorm Spark correctness regression

## Saved CPU baseline and GPU-only regression

`sqlstorm.regression` has two explicit modes: manually run `cpu-baseline` once
to save complete typed results, schemas and query times; subsequently run
`gpu-regression` against that saved directory. GPU mode never launches CPU or
rebuilds a missing/stale reference. The baseline is immutable. Changing SQL,
data files, SQL configuration, Spark version or CPU resource settings requires
an explicitly selected new reference. Empty results are valid comparisons.
There is no GPU operator coverage gate.

The initial `suites/nds-smoke.json` and `suites/nds-h-smoke.json` contain 12
queries each, using Spark 4.0.4 and SF1 Parquet. They include unchanged queries
and conservative adaptations: redundant nonrecursive `RECURSIVE`, literal
FETCH/OFFSET syntax, and unsized VARCHAR/TEXT casts. Original SQL bytes, MIT
license, original corpus cardinality and both executable/source hashes remain
in the bundle. These are small scheduled smoke sets, not full-corpus coverage.
NDS-H queries 6 and 17 exceed the complete-result smoke row bound on SF1 and
remain larger-output campaign candidates; they are not passing smoke cases.

Use Python 3.11+ for the pinned DuckDB reference engine. Install the pinned
adapter/reference dependencies, freeze the original corpus,
then build a selected adapted bundle:

```bash
python3 -m pip install -r sqlstorm/requirements.txt
python3 -m sqlstorm.bundle --source /path/to/SQLStorm --dataset nds \
  --all --output /new/original-bundle
python3 -m sqlstorm.adapt --bundle /new/original-bundle --output /new/bundle \
  --select 16,30,37,47,109,2093,3109,6191,928,9141,26859,21689
python3 -m sqlstorm.verify --bundle /new/bundle --data /data/nds-sf1 --capture-mib 32
python3 -m sqlstorm.regression --mode cpu-baseline --bundle /new/bundle \
  --data /data/nds-sf1 --baseline /saved/nds-spark404-v1 \
  --spark-home /path/to/spark-4.0.4 --python /path/to/python3.11 \
  --output /new/cpu-run --capture-mib 32
python3 -m sqlstorm.regression --mode gpu-regression --bundle /new/bundle \
  --data /data/nds-sf1 --baseline /saved/nds-spark404-v1 \
  --spark-home /path/to/spark-4.0.4 --python /path/to/python3.11 \
  --plugin-jar /path/to/scala-2.13-plugin.jar --output /new/gpu-run --capture-mib 32
```

`verify` is a one-time source/adapted result comparison in DuckDB on the same
input, including logical schemas, duplicate rows and exact values. Unchanged
SQL is recorded as byte identity. Its proof is required before creating an
adapted CPU baseline. This checks the rewrite on that fixed input, not general
SQL equivalence across engines or data. GPU runs reuse the saved proof without
executing reference SQL. True recursive CTEs, unordered string aggregation and
CHAR-width semantics have no automatic rewrite; ordering review remains in
suite metadata. The adapter preserves nontarget bytes rather than regenerating
the entire SQL through a transpiler.

Each regression is bounded to 4 CPU cores, a 4 GiB driver heap (8 GiB optional),
30% GPU pool, 100,000 complete result rows and at most 32 MiB per capture.
The default is 60 seconds per query and 1,200 seconds per Spark run. Exceeding
a capture budget cannot pass; results are never sampled or truncated to match.
The runner preserves exact typed multisets, schemas, journals, logs and JUnit.
Exit 0 means every selected query matched (or every CPU baseline query
succeeded); 1 means completed execution/comparison failed; 2 means incomplete.
A CPU reference is published only after all selected queries succeed.

Speedup is saved CPU query time / current GPU query time. Both include SQL
planning and complete result retrieval, excluding measured Python encoding and
capture-file writing. This is a historical SF1 diagnostic, with single-run and
cache variance, rather than a controlled performance benchmark or a 3 TB NDS
performance claim. No speedup threshold is enforced initially.

The Blossom `raplab/Jenkinsfile.sqlstorm-regression` uses `ci_regression.py` to
stage the same code and Python 3.11 wheels on u13, using a dedicated portable
Python 3.11 runtime without changing the system interpreter. CPU mode performs verification
and baseline creation once. GPU mode stages only the GPU candidate and reads
the explicit reference. The job holds `SPARK2A_CLUSTER`, records the runner
revision and plugin URL/hash, and archives complete evidence. Weekly GPU-only
smoke can be enabled after a successful baseline and GPU validation run.
Neither the existing distributed NDS job nor its 3 TB data is modified.

## Other diagnostic entry points

## GPU execution sweep

The sweep also accepts explicitly adapted bundles from `sqlstorm.adapt`.
It validates executable and original hashes, keeps the complete original corpus
denominator, and attempts every selected query including those flagged for
semantic review. Each result identifies both original and executed SQL hashes.
Use `--expected-spark-version 4.0.4` to verify the observed runtime. An adapted
execution success is still not a CPU/GPU correctness verdict or rewrite proof;
those comparisons are separate and never inferred from execution status.

The separate `sqlstorm.sweep` entry point executes the complete pinned NDS
(15,242 queries) or NDS-H (17,036 queries) collection in bounded batches. It
preserves original SQL, including queries marked for review, and performs no
Spark compatibility screening or GPU-operator coverage gate. It runs only the
requested engine; GPU is the default. CPU is available for focused failure
reproduction, not automatically launched during the sweep.

```bash
python3 -m sqlstorm.bundle --source /path/to/SQLStorm \
  --output /new/nds-bundle --dataset nds --all
python3 -m sqlstorm.sweep --bundle /new/nds-bundle --data /data/nds-sf1 \
  --spark-home /path/to/spark --plugin-jar /path/to/plugin.jar \
  --python /path/to/python3.10 --output /new/nds-campaign
```

For NDS-H, use `--dataset nds-h` when bundling and the corresponding eight-table
Parquet dataset. `prepare_ndsh_duckdb` can create independent SF1 data using
DuckDB's TPCH extension, cast to the existing NDS-H schema and verify all eight
table counts. It requires `duckdb` (with the TPCH extension available) and
`pyspark` for Python schema types; it does not start a Spark JVM.

```bash
python3 -m sqlstorm.prepare_ndsh_duckdb --output /new/nds-h-sf1
```

The sweep drains `queryExecution.toRdd` and counts its already planned internal
rows. It does not use a logical `DataFrame.count()`, which can prune SELECT
expressions. Complete results are not collected or stored in this execution
mode, so successful execution is explicitly **not a correctness verdict or a
CPU/GPU speedup measurement**. A separate reference comparison is needed for
silent value corruption.

The default is 50 queries per Spark application and 30 seconds per query.
Timeout/crash supervision restarts the isolated worker and continues with the
remaining queries. Every finished attempt is recorded in batch journals and
`results.jsonl`; `summary.json` separates successful execution, errors,
timeouts, crashes and unexecuted queries. SQL errors and resource timeouts
require triage and are not automatically classified as product bugs. The
campaign retains source, SQL, hashes, logs and metadata; completed batch
shuffle scratch is removed from the campaign's own directory.

## Historical paired pilot

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
