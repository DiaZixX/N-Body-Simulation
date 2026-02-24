# Commands Reference

Complete reference for all CLI commands and options.

---

## Build Commands

```bash
# CPU only (no CUDA required)
cargo build --release

# GPU mode
cargo build --release --features cuda

# 3D mode (CPU)
cargo build --release --features vec3

# 3D mode (GPU)
cargo build --release --features "cuda vec3"
```

The binary is produced at `target/release/n-body-simulation`.

---

## Simulation Commands

### gui

Run the simulation with a graphical interface.

```
cargo run --release [--features cuda] [gui] [OPTIONS]
```

Invoking without any subcommand defaults to `gui` with default parameters.

| Option | Type | Default | Description |
|---|---|---|---|
| `-n`, `--num-bodies` | usize | 100,000 | Number of bodies |
| `--dt` | f32 | 0.05 | Integration time step |
| `--theta` | f32 | 1.0 | Barnes-Hut opening angle. Lower = more accurate, slower |
| `--epsilon` | f32 | 1.0 | Softening length. Prevents singularities at close range |
| `--direct` | flag | off | Use N-squared method instead of Barnes-Hut |

Examples:

```bash
# Default GUI (Barnes-Hut, 100k bodies)
cargo run --release --features cuda

# N-squared with 10,000 bodies
cargo run --release --features cuda gui --direct -n 10000

# High accuracy Barnes-Hut
cargo run --release --features cuda gui --theta 0.5 --epsilon 0.3
```

---

### headless

Run the simulation without graphics. Useful for benchmarking, profiling, and batch experiments.

```
cargo run --release [--features cuda] headless [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `-n`, `--num-bodies` | usize | 10,000 | Number of bodies |
| `-s`, `--num-steps` | usize | 100 | Number of simulation steps |
| `--dt` | f32 | 0.05 | Integration time step |
| `--theta` | f32 | 1.0 | Barnes-Hut opening angle |
| `--epsilon` | f32 | 1.0 | Softening length |
| `-e`, `--energy-interval` | usize | 10 | Print energy statistics every N steps |
| `--direct` | flag | off | Use N-squared method instead of Barnes-Hut |
| `--no-progress` | flag | off | Suppress the progress bar |

Examples:

```bash
# Barnes-Hut, 100k bodies, 100 steps, GPU
cargo run --release --features cuda headless -n 100000 -s 100 --no-progress

# N-squared, 1k bodies, 50 steps, CPU
cargo run --release headless -n 1000 -s 50 --direct

# Suppress all output except energy
cargo run --release --features cuda headless -n 10000 -s 200 -e 10 --no-progress

# 3D mode
cargo run --release --features "cuda vec3" headless -n 50000 -s 100 --no-progress
```

---

## Benchmark

The benchmark is run via a shell script, not a subcommand, because it needs to compile and run both the CPU-only and GPU-enabled binaries in sequence.

### Running the benchmark

```bash
# From the project root
chmod +x benchmark/benchmark.sh
./benchmark/benchmark.sh
```

The script:
1. Compiles the CPU binary (`cargo build --release`).
2. Compiles the GPU binary (`cargo build --release --features cuda`).
3. For each N in `(10, 100, 1000, 10000, 100000)`, runs four measurements:
   - CPU N-squared
   - CPU Barnes-Hut
   - GPU N-squared
   - GPU Barnes-Hut
4. Writes all results to `benchmark/benchmark_results.csv`.

N-squared is skipped for N > 10,000. For N between 1,000 and 10,000, fewer steps are run and the result is scaled proportionally to the reference step count.

### Visualizing results

Open `benchmark/benchmark_viz.html` in any browser (no server required). Drag and drop `benchmark_results.csv` onto the page, or paste the CSV content directly. The visualizer shows:

- Execution time as a function of N, linear and logarithmic scale
- GPU speedup over CPU for each algorithm
- Barnes-Hut gain over N-squared for each backend
- Full results table with speedup badges

### CSV format

```
n,algorithm,backend,time_ms
10,nsquare,cpu,1.2
10,barnes-hut,cpu,0.9
10,nsquare,gpu,48.3
10,barnes-hut,gpu,51.7
...
```

`NaN` is used for skipped measurements so the visualizer can handle missing data without crashing.

### Configuring the benchmark script

Edit the variables at the top of `benchmark/benchmark.sh`:

| Variable | Default | Description |
|---|---|---|
| `STEPS` | 100 | Reference step count for all measurements |
| `N_VALUES` | (10 100 1000 10000 100000) | Body counts to benchmark |
| `SKIP_N2_ABOVE` | 10000 | Skip N-squared above this N |
| `N2_STEPS_LARGE` | 5 | Steps to run for N-squared when N >= 10000 |
| `N2_STEPS_MEDIUM` | 20 | Steps to run for N-squared when N >= 1000 |

---

## Feature Flags

| Flag | Effect |
|---|---|
| `cuda` | Enable GPU backend via CUDA. Requires CUDA toolkit and compatible GPU |
| `vec2` | 2D simulation (default) |
| `vec3` | 3D simulation |

Feature flags are passed to Cargo:

```bash
cargo build --release --features cuda
cargo build --release --features "cuda vec3"
```

Only one of `vec2` or `vec3` should be active at a time. The CUDA files are conditionally compiled with `-DVEC2` or `-DVEC3` preprocessor definitions set by `build.rs`.
