#!/usr/bin/env bash
#
# Run three LMCache CTE concurrency/inflight experiments:
#   1. Fixed concurrency 4, sweep max-inflight.
#   2. Fixed max-inflight 4, sweep concurrency.
#   3. Hold effective inflight at 16 with different concurrency/inflight pairs.
#
# Each case gets an independent JSON result and debug log. All results are
# collected in one CSV for comparison.

set -u
set -o pipefail

readonly BENCHMARK="/opt/lmcache/benchmarks/storage_backend_io/storage_backend_io_benchmark.py"
readonly SERVER_CONF="/workspace/context-transfer-engine/llm-hooks/lmcache/test/lmcache_cte_config.yaml"
readonly OUTPUT_DIR="/tmp/lmcache_bench_results/cte_concurrency_inflight_sweep"
readonly SUMMARY_CSV="${OUTPUT_DIR}/summary.csv"
readonly NUM_OPS=128
readonly CHUNK_SIZE=256

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' \
  "scenario,concurrency,max_inflight,effective_inflight,status,write_ops_per_sec,read_ops_per_sec,total_elapsed_sec,integrity_passed,result_json,log_file" \
  > "${SUMMARY_CSV}"

record_result() {
  local scenario=$1
  local concurrency=$2
  local inflight=$3
  local status=$4
  local result_json=$5
  local log_file=$6

  python -c '
import csv
import json
import os
import sys

scenario, concurrency, inflight, status, result_path, log_path, summary = sys.argv[1:]
row = {
    "scenario": scenario,
    "concurrency": int(concurrency),
    "max_inflight": int(inflight),
    "effective_inflight": int(concurrency) * int(inflight),
    "status": int(status),
    "write_ops_per_sec": "",
    "read_ops_per_sec": "",
    "total_elapsed_sec": "",
    "integrity_passed": "",
    "result_json": result_path,
    "log_file": log_path,
}
if int(status) == 0 and os.path.exists(result_path):
    with open(result_path, encoding="utf-8") as stream:
        result = json.load(stream)[0]
    for key in (
        "write_ops_per_sec",
        "read_ops_per_sec",
        "total_elapsed_sec",
        "integrity_passed",
    ):
        row[key] = result.get(key, "")
with open(summary, "a", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=row)
    writer.writerow(row)
print(
    f"Finished {scenario} c={concurrency} i={inflight}: status={status}, "
    f"write={row['"'"'write_ops_per_sec'"'"']}, read={row['"'"'read_ops_per_sec'"'"']}, "
    f"integrity={row['"'"'integrity_passed'"'"']}"
)
' "${scenario}" "${concurrency}" "${inflight}" "${status}" \
    "${result_json}" "${log_file}" "${SUMMARY_CSV}"
}

run_case() {
  local scenario=$1
  local concurrency=$2
  local inflight=$3
  local scenario_dir="${OUTPUT_DIR}/${scenario}"
  local case_name="cte_c${concurrency}_i${inflight}"
  local result_json="${scenario_dir}/${case_name}.json"
  local log_file="${scenario_dir}/${case_name}.log"
  local tag_name="lmcache_${scenario}_c${concurrency}_i${inflight}"

  mkdir -p "${scenario_dir}"
  printf 'Running %s: concurrency=%s max_inflight=%s effective=%s\n' \
    "${scenario}" "${concurrency}" "${inflight}" \
    "$((concurrency * inflight))"

  CLIO_BATCH_LANE=1 CLIO_CTE_BATCHING=1 \
  CTP_LOG_LEVEL=info \
  CLIO_WITH_RUNTIME=1 \
  CLIO_BIND_ADDR=127.0.0.1 \
  CLIO_SERVER_CONF="${SERVER_CONF}" \
  CLIO_IPC_MODE=SHM \
  CLIO_REPO_PATH=/usr/local/lib \
  PYTHONPATH="/workspace/build/bin:/workspace:/opt/lmcache" \
    python "${BENCHMARK}" \
      --backend clio_cte \
      --num-ops "${NUM_OPS}" \
      --concurrency "${concurrency}" \
      --chunk-size "${CHUNK_SIZE}" \
      --write_bench False \
      --clio-cte-tag-name "${tag_name}" \
      --clio-cte-pool-query-mode local \
      --clio-cte-max-inflight "${inflight}" \
      --output-json "${result_json}" \
      --verify-integrity \
      > "${log_file}" 2>&1
  local status=$?

  record_result "${scenario}" "${concurrency}" "${inflight}" "${status}" \
    "${result_json}" "${log_file}"
}

run_inflight_sweep() {
  local inflight
  for inflight in 1 2 4 8 16; do
    run_case "fixed_concurrency_4" 4 "${inflight}"
  done
}

run_concurrency_sweep() {
  local concurrency
  for concurrency in 1 2 4 8 16; do
    run_case "fixed_inflight_4" "${concurrency}" 4
  done
}

run_fixed_effective_sweep() {
  local pair concurrency inflight
  for pair in "1 16" "2 8" "4 4" "8 2" "16 1"; do
    read -r concurrency inflight <<< "${pair}"
    run_case "fixed_effective_16" "${concurrency}" "${inflight}"
  done
}

run_inflight_sweep
run_concurrency_sweep
run_fixed_effective_sweep

printf 'Sweep complete: %s\n' "${SUMMARY_CSV}"
