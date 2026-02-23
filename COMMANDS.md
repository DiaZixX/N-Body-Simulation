# N-Body Simulation - Commands Guide

## Table of Contents

1. [Quick Start Commands](#quick-start-commands)
2. [Graphical Mode (GUI)](#graphical-mode-gui)
3. [Headless Mode](#headless-mode)
4. [Detailed Parameters](#detailed-parameters)
5. [Summary Tables](#summary-tables)
6. [Usage Examples](#usage-examples)
7. [Benchmarking](#benchmarking)
8. [Compilation](#compilation)

---

## Quick Start Commands

### Ultra-Fast Startup

```bash
# Launch with graphical interface (default)
cargo run --release

# Launch a headless simulation
cargo run --release headless

# High-performance GPU version
cargo run --release --features cuda
```

---

## Graphical Mode (GUI)

### General Syntax

```bash
cargo run --release [FEATURES] gui [OPTIONS]
```

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-n, --num-bodies` | usize | 100000 | Number of particles |
| `--dt` | f32 | 0.05 | Time step (integration) |
| `--theta` | f32 | 1.0 | Barnes-Hut theta parameter (accuracy) |
| `--epsilon` | f32 | 1.0 | Softening parameter |
| `--direct` | flag | false | Use N² instead of Barnes-Hut |

### GUI Examples

#### Default Configuration

```bash
# 2D CPU Barnes-Hut, 100k bodies
cargo run --release

# 3D CPU Barnes-Hut, 100k bodies
cargo run --release --no-default-features --features vec3

# 2D GPU Barnes-Hut, 100k bodies
cargo run --release --features cuda

# 3D GPU Barnes-Hut, 100k bodies
cargo run --release --no-default-features --features "vec3,cuda"
```

#### Customizing Number of Particles

```bash
# Small simulation (10k bodies)
cargo run --release gui -n 10000

# Medium simulation (500k bodies, GPU recommended)
cargo run --release --features cuda gui -n 500000

# Large simulation (1M bodies, GPU required)
cargo run --release --features cuda gui -n 1000000

# 3D with 200k bodies
cargo run --release --no-default-features --features "vec3,cuda" gui -n 200000
```

#### Adjusting Precision

```bash
# High precision (small theta, small dt)
cargo run --release gui -n 50000 --theta 0.3 --dt 0.01

# Maximum precision with direct N²
cargo run --release gui -n 5000 --direct --dt 0.001

# GPU high precision
cargo run --release --features cuda gui -n 100000 --theta 0.5 --dt 0.02
```

#### Adjusting Softening

```bash
# High softening (soft collisions)
cargo run --release gui -n 100000 --epsilon 2.0

# Low softening (precise interactions)
cargo run --release --features cuda gui -n 100000 --epsilon 0.5

# No softening (beware of singularities)
cargo run --release gui -n 10000 --epsilon 0.1
```

#### Direct N² Method (Exact)

```bash
# 2D CPU N² (small number recommended)
cargo run --release gui -n 5000 --direct

# 2D GPU N² (can handle more particles)
cargo run --release --features cuda gui -n 50000 --direct

# 3D GPU N²
cargo run --release --no-default-features --features "vec3,cuda" gui -n 30000 --direct
```

---

## Headless Mode

### General Syntax

```bash
cargo run --release [FEATURES] headless [OPTIONS]
```

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-n, --num-bodies` | usize | 10000 | Number of particles |
| `-s, --num-steps` | usize | 100 | Number of simulation steps |
| `--dt` | f32 | 0.05 | Time step |
| `--theta` | f32 | 1.0 | Barnes-Hut theta parameter |
| `--epsilon` | f32 | 1.0 | Softening parameter |
| `-e, --energy-interval` | usize | 10 | Energy display interval |
| `--direct` | flag | false | Use N² instead of Barnes-Hut |
| `--no-progress` | flag | false | Disable progress bar |

### Headless Examples

#### Basic Simulations

```bash
# Default configuration (10k bodies, 100 steps)
cargo run --release headless

# Customized
cargo run --release headless -n 50000 -s 500

# GPU
cargo run --release --features cuda headless -n 500000 -s 200
```

#### Energy Conservation Tests

```bash
# CPU Barnes-Hut, frequent energy display
cargo run --release headless -n 10000 -s 1000 --dt 0.001 -e 10

# GPU N² (exact reference)
cargo run --release --features cuda headless -n 10000 -s 1000 --direct --dt 0.001 -e 1

# Method comparison
cargo run --release headless -n 10000 -s 500 -e 5
cargo run --release headless -n 10000 -s 500 --direct -e 5
```

#### Long Simulations

```bash
# 1M steps with GPU
cargo run --release --features cuda headless -n 100000 -s 1000 -e 50

# 3D long duration
cargo run --release --no-default-features --features "vec3,cuda" headless -n 200000 -s 5000 -e 100
```

#### Without Progress Bar (for redirection)

```bash
# Redirect to file
cargo run --release headless -n 100000 -s 1000 --no-progress > results.txt

# Pipeline with grep
cargo run --release --features cuda headless -n 500000 -s 500 --no-progress | grep "E_total"
```

---

## Detailed Parameters

### `num_bodies` (Number of Particles)

| Value | Recommendation | Backend |
|-------|----------------|---------|
| 1K - 10K | CPU N² | CPU |
| 10K - 100K | CPU Barnes-Hut or GPU N² | CPU/GPU |
| 100K - 1M | GPU Barnes-Hut | GPU |
| > 1M | GPU Barnes-Hut high perf | GPU |

### `dt` (Time Step)

| Value | Usage | Precision | Performance |
|-------|-------|-----------|-------------|
| 0.001 - 0.01 | Energy conservation | 5 stars | 1 star |
| 0.01 - 0.05 | Balance | 4 stars | 3 stars |
| 0.05 - 0.1 | Fast visualization | 3 stars | 5 stars |
| > 0.1 | Risk of instability | 1 star | 5 stars |

### `theta` (Barnes-Hut Parameter)

| Value | Precision | Performance | Usage |
|-------|-----------|-------------|-------|
| 0.3 - 0.5 | 5 stars | 2 stars | Scientific simulations |
| 0.5 - 0.7 | 4 stars | 3 stars | Balance precision/speed |
| 0.7 - 1.0 | 3 stars | 4 stars | Real-time visualizations |
| > 1.0 | 2 stars | 5 stars | Fast demos |

**Note:** Smaller `theta` values give more accurate but slower simulations.

### `epsilon` (Softening)

| Value | Effect | Usage |
|-------|--------|-------|
| 0.1 - 0.5 | Strong interactions | Planetary systems |
| 0.5 - 1.0 | Balance | General use |
| 1.0 - 2.0 | Soft collisions | Galaxies, clusters |
| > 2.0 | Very smooth | Large scales |

**Note:** Prevents singularities during close collisions.

### `energy_interval`

| Value | Usage |
|-------|-------|
| 1 | Precise conservation test |
| 10 | Standard monitoring |
| 50-100 | Long simulations |

---

## Summary Tables

### GUI Mode

| Dimension | Method | Backend | Num Bodies | Command |
|-----------|--------|---------|------------|---------|
| **2D** | Barnes-Hut | CPU | 10K - 100K | `cargo run --release gui -n 100000` |
| **2D** | Barnes-Hut | GPU | 100K - 1M | `cargo run --release --features cuda gui -n 500000` |
| **2D** | N² | CPU | 1K - 10K | `cargo run --release gui -n 5000 --direct` |
| **2D** | N² | GPU | 10K - 100K | `cargo run --release --features cuda gui -n 50000 --direct` |
| **3D** | Barnes-Hut | CPU | 10K - 50K | `cargo run --release --no-default-features --features vec3 gui -n 50000` |
| **3D** | Barnes-Hut | GPU | 50K - 500K | `cargo run --release --no-default-features --features "vec3,cuda" gui -n 200000` |
| **3D** | N² | CPU | 1K - 5K | `cargo run --release --no-default-features --features vec3 gui -n 3000 --direct` |
| **3D** | N² | GPU | 5K - 50K | `cargo run --release --no-default-features --features "vec3,cuda" gui -n 30000 --direct` |

### Headless Mode

| Dimension | Method | Backend | Command Type |
|-----------|--------|---------|--------------|
| **2D** | Barnes-Hut | CPU | `cargo run --release headless -n 100000 -s 500` |
| **2D** | Barnes-Hut | GPU | `cargo run --release --features cuda headless -n 1000000 -s 200` |
| **2D** | N² | CPU | `cargo run --release headless -n 10000 -s 500 --direct` |
| **2D** | N² | GPU | `cargo run --release --features cuda headless -n 100000 -s 200 --direct` |
| **3D** | Barnes-Hut | CPU | `cargo run --release --no-default-features --features vec3 headless -n 50000 -s 500` |
| **3D** | Barnes-Hut | GPU | `cargo run --release --no-default-features --features "vec3,cuda" headless -n 500000 -s 200` |
| **3D** | N² | CPU | `cargo run --release --no-default-features --features vec3 headless -n 5000 -s 500 --direct` |
| **3D** | N² | GPU | `cargo run --release --no-default-features --features "vec3,cuda" headless -n 50000 -s 200 --direct` |

---

## Usage Examples

### Use Case 1: Quick Demonstration

**Goal:** Show a visually impressive simulation

```bash
# GPU with many particles
cargo run --release --features cuda gui -n 500000 --dt 0.05

# 3D for more visual impact
cargo run --release --no-default-features --features "vec3,cuda" gui -n 300000
```

### Use Case 2: Scientific Validation

**Goal:** Verify energy conservation

```bash
# Exact N² with small dt
cargo run --release headless -n 5000 -s 2000 --direct --dt 0.001 -e 1

# Compare with Barnes-Hut
cargo run --release headless -n 5000 -s 2000 --theta 0.3 --dt 0.001 -e 1
```

### Use Case 3: Maximum Performance

**Goal:** Simulate as many particles as possible

```bash
# GPU 2D with 1M particles
cargo run --release --features cuda gui -n 1000000 --theta 1.0 --dt 0.05

# Headless for benchmark
cargo run --release --features cuda headless -n 2000000 -s 100 --no-progress
```

### Use Case 4: Maximum Precision

**Goal:** Most accurate results possible

```bash
# N² with very small dt
cargo run --release gui -n 3000 --direct --dt 0.0005 --epsilon 0.1

# Precise Barnes-Hut
cargo run --release gui -n 20000 --theta 0.3 --dt 0.001 --epsilon 0.5
```

### Use Case 5: Planetary Systems

**Goal:** Simulate a solar system

```bash
# Small epsilon, adapted dt
cargo run --release gui -n 10 --dt 0.01 --epsilon 0.5 --direct
```

### Use Case 6: Galaxy Formation

**Goal:** Large systems with many particles

```bash
# GPU with high softening
cargo run --release --features cuda gui -n 500000 --epsilon 2.0 --dt 0.05
```

---

## Benchmarking

### CPU vs GPU Comparison

#### 2D - 100K Bodies - 100 Steps

```bash
# CPU Barnes-Hut
time cargo run --release headless -n 100000 -s 100 --no-progress

# GPU Barnes-Hut
time cargo run --release --features cuda headless -n 100000 -s 100 --no-progress
```

**Expected Result:** GPU ~10-50x faster depending on configuration

#### 3D - 50K Bodies - 100 Steps

```bash
# CPU
time cargo run --release --no-default-features --features vec3 headless -n 50000 -s 100 --no-progress

# GPU
time cargo run --release --no-default-features --features "vec3,cuda" headless -n 50000 -s 100 --no-progress
```

### N² vs Barnes-Hut Comparison

#### CPU - 10K Bodies

```bash
# Direct N²
time cargo run --release headless -n 10000 -s 100 --direct --no-progress

# Barnes-Hut
time cargo run --release headless -n 10000 -s 100 --no-progress
```

**Expected Result:** Barnes-Hut ~2-5x faster

#### GPU - 100K Bodies

```bash
# Direct N²
time cargo run --release --features cuda headless -n 100000 -s 50 --direct --no-progress

# Barnes-Hut
time cargo run --release --features cuda headless -n 100000 -s 50 --no-progress
```

**Expected Result:** Barnes-Hut ~5-10x faster

### Scalability

#### Testing Different Sizes

```bash
# Create a benchmark script
for n in 1000 10000 100000 500000 1000000; do
    echo "Testing $n bodies..."
    time cargo run --release --features cuda headless -n $n -s 50 --no-progress
done
```

### Complete Benchmark

```bash
# Script to test all combinations
#!/bin/bash

methods=("" "--direct")
backends=("" "--features cuda")
dims=("" "--no-default-features --features vec3")
sizes=(1000 10000 100000)

for method in "${methods[@]}"; do
  for backend in "${backends[@]}"; do
    for dim in "${dims[@]}"; do
      for size in "${sizes[@]}"; do
        echo "Testing: $size bodies, $method $backend $dim"
        time cargo run --release $backend $dim headless -n $size -s 10 $method --no-progress
      done
    done
  done
done
```

---

## Compilation

### Basic Builds

```bash
# 2D CPU (default)
cargo build --release

# 3D CPU
cargo build --release --no-default-features --features vec3

# 2D GPU
cargo build --release --features cuda

# 3D GPU
cargo build --release --no-default-features --features "vec3,cuda"
```

### Cleaning

```bash
# Clean compilation artifacts
cargo clean

# Clean and rebuild
cargo clean && cargo build --release
```

### Verification

```bash
# Check compilation without building
cargo check

# Check with CUDA
cargo check --features cuda
```

### Additional Optimizations

For maximum performance, create a profile in `Cargo.toml`:

```toml
[profile.max-perf]
inherits = "release"
lto = true
codegen-units = 1
```

Then compile with:

```bash
cargo build --profile max-perf --features cuda
cargo run --profile max-perf --features cuda
```

---

## Help and Documentation

### Getting Help

```bash
# General help
cargo run --release -- --help

# GUI help
cargo run --release gui --help

# Headless help
cargo run --release headless --help
```

### Useful Environment Variables

```bash
# Backtrace on error
RUST_BACKTRACE=1 cargo run --release

# Full backtrace
RUST_BACKTRACE=full cargo run --release

# Detailed logging
RUST_LOG=debug cargo run --release
```

### CUDA Verification

```bash
# Check CUDA availability
nvcc --version

# Check available GPUs
nvidia-smi
```

---

## Important Notes

### Limitations

- **Maximum particles (GUI):** ~10M (depends on VRAM)
- **Maximum particles (Headless):** ~100M (depends on RAM)
- **GPU Barnes-Hut:** Limited to 1M nodes in tree (MAX_NODES)

### Known Issues

1. **Tree node limit reached:** Reduce `num_bodies` or increase `MAX_NODES` in `barnes_hut.cu`
2. **Out of memory:** Reduce `num_bodies` or use fewer particles
3. **Unstable simulation:** Reduce `dt` or increase `epsilon`

### Performance Tips

- **GUI:** Prefer GPU for >100K particles
- **Headless:** Use `--no-progress` for clean benchmarks
- **Energy conservation:** Use `dt < 0.01` and `theta < 0.5`
- **Real-time visualization:** Use `theta = 1.0` and `dt = 0.05`

---

**Version:** 1.0.0  
**Last Updated:** 2024  
**Author:** N-Body Simulation Project
