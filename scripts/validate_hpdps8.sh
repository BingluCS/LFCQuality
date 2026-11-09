#!/usr/bin/env bash
# Explicitly enforce the project's current validation host.
set -euo pipefail
if [[ $(hostname -s) != HPDPS-8 ]]; then
  echo 'Validation must run on HPDPS-8.' >&2
  exit 2
fi
cd "$(dirname "$0")/.."
quality_python=${LFCQUALITY_PYTHON:-"$PWD/.venv/bin/python"}
quality_report=${1:-"results/hpdps8-$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$(dirname "$quality_report")"
mkdir "$quality_report"
quality_run=(numactl --physcpubind="${LFCQUALITY_CPUS:-32-39}" --membind="${LFCQUALITY_NODE:-1}")
export OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
if [[ -d .native-deps ]]; then
  export LFCQUALITY_FFTW_PREFIX="$PWD/.native-deps"
fi
hostname > "$quality_report/host.txt"
lscpu > "$quality_report/lscpu.txt"
"${quality_run[@]}" numactl --show > "$quality_report/numa-binding.txt"
"${quality_run[@]}" "$quality_python" -m lfcquality build > "$quality_report/native-build.json"
quality_tests=(tests)
if [[ -n ${LFCBENCH_CHECKOUT:-} ]]; then
  quality_tests+=("$LFCBENCH_CHECKOUT/tests")
fi
"${quality_run[@]}" "$quality_python" -m pytest -q "${quality_tests[@]}" --junitxml="$quality_report/pytest.xml" | tee "$quality_report/pytest.txt"
"${quality_run[@]}" cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${LFCQUALITY_FFTW_PREFIX:-}" > "$quality_report/cmake-configure.txt"
"${quality_run[@]}" cmake --build build --parallel 4 > "$quality_report/cmake-build.txt"
"${quality_run[@]}" ctest --test-dir build --output-on-failure | tee "$quality_report/ctest.txt"
"${quality_run[@]}" "$quality_python" examples/demo.py --output "$quality_report/demo" | tee "$quality_report/demo.txt"
"${quality_run[@]}" "$quality_python" -m lfcquality compare-fields "$quality_report/demo/fields.json" --output "$quality_report/demo-cli.json"
"${quality_run[@]}" "$quality_python" -m build > "$quality_report/package-build.txt" 2>&1
echo "Validation artifacts: $quality_report"
