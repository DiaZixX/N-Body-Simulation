#!/usr/bin/env bash
# benchmark.sh — N-Body performance benchmark
#
# Compiles both CPU and GPU variants, runs N² and Barnes-Hut for each N value,
# and writes all results to a single CSV file ready for benchmark_viz.html.
#
# Usage:
#   cd <project_root>
#   chmod +x benchmark/benchmark.sh
#   ./benchmark/benchmark.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

STEPS=100
N_VALUES=(10 100 1000 10000 100000)
OUTPUT="benchmark/benchmark_results.csv"

# N² is O(N²): for large N we run fewer steps and scale the result
N2_STEPS_LARGE=5      # N >= 10000
N2_STEPS_MEDIUM=20    # N >= 1000
SKIP_N2_ABOVE=10000   # skip N² entirely above this (100000 would take hours)

# ── Colors ────────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────────────────

# Format milliseconds into a human-readable string.
# Uses only awk (always available) — no bc dependency.
fmt_ms() {
    local ms="$1"
    [[ "$ms" == "NaN" ]] && echo "skipped" && return
    awk -v ms="$ms" 'BEGIN {
        if      (ms >= 3600000) printf "%.2f h",   ms / 3600000
        else if (ms >= 60000)   printf "%.2f min",  ms / 60000
        else if (ms >= 1000)    printf "%.2f s",    ms / 1000
        else                    printf "%d ms",     ms
    }'
}

# Run one headless simulation and return elapsed time in ms.
# Args: binary  n  steps  [extra args...]
run_sim() {
    local bin="$1" n="$2" steps="$3"; shift 3
    local t_start t_end

    t_start=$(date +%s%N)
    if "$bin" headless -n "$n" -s "$steps" --no-progress -e $((steps + 1)) "$@" &>/dev/null; then
        t_end=$(date +%s%N)
        echo $(( (t_end - t_start) / 1000000 ))
    else
        echo "NaN"
    fi
}

# Scale a raw time measured over actual_steps to the full STEPS equivalent.
# Uses awk for float-safe arithmetic — avoids integer overflow on large values.
scale_time() {
    local raw="$1" actual="$2"
    [[ "$raw" == "NaN" ]] && echo "NaN" && return
    awk -v raw="$raw" -v actual="$actual" -v steps="$STEPS" \
        'BEGIN { printf "%d", int(raw * steps / actual + 0.5) }'
}

# ── Build ─────────────────────────────────────────────────────────────────────

build() {
    local label="$1"; shift
    echo -ne "  Building ${BOLD}${label}${RESET} ... "
    if cargo build --release "$@" 2>/dev/null; then
        echo -e "${GREEN}OK${RESET}"
        return 0
    else
        echo -e "${RED}FAILED${RESET}"
        return 1
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────

mkdir -p benchmark

echo -e "${BOLD}════════════════════════════════════════${RESET}"
echo -e "${BOLD}  N-Body Benchmark${RESET}"
echo -e "  Steps   : ${CYAN}${STEPS}${RESET}"
echo -e "  N values: ${CYAN}${N_VALUES[*]}${RESET}"
echo -e "  Output  : ${CYAN}${OUTPUT}${RESET}"
echo -e "${BOLD}════════════════════════════════════════${RESET}"
echo

echo -e "${BOLD}── Build ───────────────────────────────${RESET}"

CPU_BIN="" GPU_BIN=""

if build "CPU (no --features cuda)"; then
    CPU_BIN=/tmp/nbody_cpu_bench
    cp ./target/release/n-body-simulation "$CPU_BIN"
fi

if build "GPU (--features cuda)" --features cuda; then
    GPU_BIN=/tmp/nbody_gpu_bench
    cp ./target/release/n-body-simulation "$GPU_BIN"
fi

echo

# Write CSV header
echo "n,algorithm,backend,time_ms" > "$OUTPUT"

# ── Measurements ──────────────────────────────────────────────────────────────

for n in "${N_VALUES[@]}"; do
    echo -e "${BOLD}── N = ${CYAN}${n}${RESET}${BOLD} ──────────────────────────────${RESET}"

    # How many steps to actually run for N² (fewer for large N, then scale)
    if   (( n >= 10000 )); then n2_steps=$N2_STEPS_LARGE
    elif (( n >= 1000  )); then n2_steps=$N2_STEPS_MEDIUM
    else                        n2_steps=$STEPS
    fi

    skip_n2=false
    (( n > SKIP_N2_ABOVE )) && skip_n2=true

    # ── CPU N² ──────────────────────────────────────────────────────────────
    if [[ -n "$CPU_BIN" ]] && ! $skip_n2; then
        printf "  %-18s" "CPU N²"
        raw=$(run_sim "$CPU_BIN" "$n" "$n2_steps" --direct)
        t=$(scale_time "$raw" "$n2_steps")
        echo -e ": ${CYAN}$(fmt_ms "$t")${RESET}  (measured ${n2_steps} steps, scaled to ${STEPS})"
    else
        t="NaN"
        if $skip_n2; then
            printf "  %-18s: ${YELLOW}skipped (N > %d)${RESET}\n" "CPU N²" "$SKIP_N2_ABOVE"
        else
            printf "  %-18s: ${RED}skipped (build failed)${RESET}\n" "CPU N²"
        fi
    fi
    echo "${n},nsquare,cpu,${t}" >> "$OUTPUT"

    # ── CPU Barnes-Hut ───────────────────────────────────────────────────────
    if [[ -n "$CPU_BIN" ]]; then
        printf "  %-18s" "CPU Barnes-Hut"
        t=$(run_sim "$CPU_BIN" "$n" "$STEPS")
        echo -e ": ${CYAN}$(fmt_ms "$t")${RESET}"
    else
        t="NaN"
        printf "  %-18s: ${RED}skipped (build failed)${RESET}\n" "CPU Barnes-Hut"
    fi
    echo "${n},barnes-hut,cpu,${t}" >> "$OUTPUT"

    # ── GPU N² ──────────────────────────────────────────────────────────────
    if [[ -n "$GPU_BIN" ]] && ! $skip_n2; then
        printf "  %-18s" "GPU N²"
        raw=$(run_sim "$GPU_BIN" "$n" "$n2_steps" --direct)
        t=$(scale_time "$raw" "$n2_steps")
        echo -e ": ${CYAN}$(fmt_ms "$t")${RESET}  (measured ${n2_steps} steps, scaled to ${STEPS})"
    else
        t="NaN"
        if $skip_n2; then
            printf "  %-18s: ${YELLOW}skipped (N > %d)${RESET}\n" "GPU N²" "$SKIP_N2_ABOVE"
        else
            printf "  %-18s: ${RED}skipped (build failed)${RESET}\n" "GPU N²"
        fi
    fi
    echo "${n},nsquare,gpu,${t}" >> "$OUTPUT"

    # ── GPU Barnes-Hut ───────────────────────────────────────────────────────
    if [[ -n "$GPU_BIN" ]]; then
        printf "  %-18s" "GPU Barnes-Hut"
        t=$(run_sim "$GPU_BIN" "$n" "$STEPS")
        echo -e ": ${CYAN}$(fmt_ms "$t")${RESET}"
    else
        t="NaN"
        printf "  %-18s: ${RED}skipped (build failed)${RESET}\n" "GPU Barnes-Hut"
    fi
    echo "${n},barnes-hut,gpu,${t}" >> "$OUTPUT"

    echo
done

# ── Done ──────────────────────────────────────────────────────────────────────

echo -e "${BOLD}════════════════════════════════════════${RESET}"
echo -e "  ${GREEN}Results written to : ${CYAN}${OUTPUT}${RESET}"
echo -e "  Open ${CYAN}benchmark/benchmark_viz.html${RESET}"
echo -e "  in a browser and load this CSV."
echo -e "${BOLD}════════════════════════════════════════${RESET}"
