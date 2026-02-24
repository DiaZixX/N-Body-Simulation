# N-Body Simulation

A gravitational N-body simulator written in Rust with optional GPU acceleration via CUDA. The simulator implements two force computation algorithms (direct N-squared and Barnes-Hut) on two backends (CPU and GPU), and supports both an interactive graphical interface and a headless batch mode.

---

## Table of Contents

1. [Overview](#overview)
2. [Algorithms](#algorithms)
3. [Architecture](#architecture)
4. [Requirements](#requirements)
5. [Building](#building)
6. [Usage](#usage)
7. [Benchmark](#benchmark)
8. [Performance Notes](#performance-notes)

---

## Overview

The simulation models a system of N point masses interacting through Newtonian gravity. Each body exerts a gravitational force on every other body, and the system evolves through numerical integration using a fixed time step (Euler or leapfrog depending on configuration).

Two simulation modes are available:

- **GUI mode**: real-time rendering with interactive controls.
- **Headless mode**: runs a fixed number of steps without graphics, used for benchmarking and batch experiments.

---

## Algorithms

### Direct N-squared (brute force)

Every pair of bodies is evaluated at each step. Computational complexity is O(N^2) per step, which makes it exact but prohibitively slow for large N. For N = 100,000, a single step requires ~5 billion pairwise evaluations.

On GPU, this algorithm benefits massively from parallelism. The kernel uses shared memory tiling: each thread block loads a tile of bodies into shared memory and computes interactions for all bodies in that tile simultaneously, reducing global memory traffic by a factor equal to the tile size.

### Barnes-Hut

The Barnes-Hut algorithm reduces complexity to O(N log N) by grouping distant bodies into a single representative mass. The simulation space is recursively subdivided into a quadtree (2D) or octree (3D). For each body, the tree is traversed: if a cell is sufficiently far away (determined by the opening angle parameter theta), the entire cell is approximated as a single body at its center of mass. Otherwise the traversal descends into the cell's children.

Key implementation details:

- **Tree construction** is performed on the CPU. The tree is inherently sequential due to data dependencies between insertions; parallelizing construction correctly requires substantially more complex algorithms (e.g., Lauterbach-Karras on GPU) which are not yet implemented.
- **Force computation** is performed on the GPU in parallel. Each thread handles one body and traverses the tree independently.
- **Skip pointers**: each tree node stores a "next" pointer that allows the traversal to skip an entire subtree when the opening criterion is met, without a recursive call stack.
- **Sentinel value**: the end of traversal is marked by `next = -1`, which avoids the ambiguity of using index 0 (the root node) as a sentinel.
- **Morton sort**: before tree construction, bodies are sorted by their Morton code (Z-order curve). Spatially adjacent bodies receive adjacent Morton indices, so neighboring GPU threads traverse nearly identical paths through the tree, dramatically reducing warp divergence.
- **Compact GPU node structure**: the full `BHNode` used for construction (40 bytes, includes cell center coordinates needed for subdivision) is converted to a compact `BHNodeGPU` (32 bytes, removes construction-only fields) before upload. This reduces PCIe transfer volume and improves cache utilization on the GPU.
- **Pinned memory**: CPU buffers for the sorted positions and the compact tree are allocated with `cudaHostAlloc` (page-locked). DMA transfers from pinned memory proceed without an intermediate staging copy, achieving close to theoretical PCIe bandwidth.

#### Opening angle (theta)

The theta parameter controls the accuracy-performance tradeoff. A cell of size `s` at distance `d` is approximated if `s / d < theta`. Lower theta means more accurate but slower simulation. The default value of 1.0 is standard for cosmological simulations; values between 0.5 and 1.5 are typical.

#### Softening parameter (epsilon)

To prevent numerical singularities when two bodies approach each other closely, the distance in the force formula is softened: the denominator uses `sqrt(r^2 + epsilon^2)` instead of `r`. Epsilon effectively sets a minimum interaction distance.

---

## Architecture

```
n-body-simulation/
  Cargo.toml
  build.rs                      Invokes nvcc to compile .cu files via cc-rs
  README.md
  COMMANDS.md
  src/
    main.rs                     Entry point, CLI parsing (clap)
    lib.rs                      Public API, re-exports
    headless.rs                 Headless simulation runner
    body/
      body.rs                   Body struct: position, velocity, acceleration, mass
      mod.rs
    geom/
      vec2.rs                   2D vector type
      vec3.rs                   3D vector type
      vector.rs                 Shared vector trait
      mod.rs
    kdtree/
      kdtree.rs                 Barnes-Hut tree: construction and CPU traversal
      kdcell.rs                 Bounding cell geometry
      node.rs                   Tree node definition
      mod.rs
    simul/
      compute.rs                N-squared CPU force kernel
      energy.rs                 Kinetic and potential energy computation
      generate.rs               Initial condition generators (uniform disc, etc.)
      stats.rs                  Performance statistics and progress reporting
      mod.rs
    renderer/
      state.rs                  wgpu render state
      app.rs                    winit event loop and simulation step integration
      camera.rs                 Camera and zoom controls
      vertex.rs                 Vertex buffer layout
      instance.rs               Per-body GPU instance data
      shader.wgsl               WGSL vertex and fragment shaders
      config.rs                 Renderer configuration
      mod.rs
    cuda/
      mod.rs                    Rust FFI bindings (extern "C" declarations)
      kernel.cu                 GPU N-squared kernel (shared memory tiling)
      barnes_hut.cu             CPU tree construction + GPU force kernel
                                (Morton sort, compact node struct, pinned memory)
  benchmark/
    benchmark.sh                Compiles CPU and GPU binaries, runs all
                                combinations, writes benchmark_results.csv
    benchmark_viz.html          Standalone HTML visualizer for the CSV results
    benchmark_results.csv       Generated output, not committed
  tests/
    kdtree_tests.rs             Unit tests for the Barnes-Hut tree
    vector_tests.rs             Unit tests for the vector types
```

### CPU vs GPU backends

The backend is selected at compile time via Cargo feature flags:

| Feature flag       | Effect                        |
|--------------------|-------------------------------|
| *(none)*           | CPU only                      |
| `--features cuda`  | GPU force kernels via CUDA    |
| `--features vec2`  | 2D simulation (default)       |
| `--features vec3`  | 3D simulation                 |

When compiled with `cuda`, both the N-squared and Barnes-Hut force computations run on the GPU. Tree construction in Barnes-Hut always runs on the CPU regardless of the feature flag. The `.cu` files are compiled by `build.rs` using `cc-rs`, which invokes `nvcc` with the appropriate `-DVEC2` or `-DVEC3` preprocessor definition.

### Data flow (Barnes-Hut GPU, one step)

```
CPU: compute bounding box                  O(N), sequential
CPU: sort bodies by Morton code            O(N), radix sort 4 passes
CPU: insert bodies into quadtree           O(N log N), sequential
CPU: propagate masses bottom-up            O(nodes), depth-first post-order
CPU: extract compact BHNodeGPU array       O(nodes)
PCIe: upload tree (pinned, async stream 1)
PCIe: upload sorted positions (pinned, async stream 2)
GPU: compute forces                        N threads in parallel, O(log N) per thread
PCIe: download accelerations
CPU: reorder accelerations to original indices
CPU: integrate positions and velocities    O(N)
```

---

- Rust 1.70 or later
- For GPU mode: CUDA toolkit (tested with CUDA 11.x and 12.x), an NVIDIA GPU with compute capability 7.5 or later (Turing architecture, RTX 2000 series and above)

---

## Building

```bash
# CPU only (no CUDA required)
cargo build --release

# GPU mode (requires CUDA toolkit and compatible GPU)
cargo build --release --features cuda

# 3D mode (CPU)
cargo build --release --features vec3

# 3D mode (GPU)
cargo build --release --features "cuda vec3"
```

---

## Usage

### GUI mode

```bash
# Default: Barnes-Hut, 100,000 bodies
cargo run --release --features cuda

# Direct N^2 method
cargo run --release --features cuda gui --direct

# Custom parameters
cargo run --release --features cuda gui -n 50000 --theta 0.8 --epsilon 0.5
```

### Headless mode

```bash
# Barnes-Hut, 100,000 bodies, 100 steps
cargo run --release --features cuda headless -n 100000 -s 100 --no-progress

# Direct N^2
cargo run --release --features cuda headless -n 10000 -s 50 --direct

# Print energy every 5 steps
cargo run --release --features cuda headless -n 10000 -s 100 -e 5
```

Full option reference: see [COMMANDS.md](COMMANDS.md).

---

## Benchmark

The benchmark script compiles both the CPU-only and GPU-enabled binaries, runs the four algorithm/backend combinations for each body count, and writes a CSV file.

```bash
chmod +x benchmark/benchmark.sh
./benchmark/benchmark.sh
```

Results are written to `benchmark/benchmark_results.csv`. Open `benchmark/benchmark_viz.html` in any browser and load the CSV to display:

- Execution time curves (linear and logarithmic scale)
- GPU speedup over CPU for each algorithm
- Barnes-Hut gain over N-squared for each backend

N-squared is skipped for N > 10,000 by default because the O(N^2) cost makes it impractical (a 100,000-body N-squared run at 100 steps would take several hours on CPU). For N between 1,000 and 10,000, the script runs fewer steps and scales the result proportionally.

---

## Performance Notes

### Why tree construction stays on the CPU

Quadtree and octree construction has sequential data dependencies: inserting body B may require subdividing a leaf that was created by body A. Parallelizing this correctly on GPU requires locking schemes or a fundamentally different algorithm (parallel BVH construction, e.g., Karras 2012). The CPU implementation in O(N log N) sequential time is fast enough that it does not dominate the total step time for N up to ~500,000.

### Warp divergence in Barnes-Hut GPU

The main bottleneck of the GPU force kernel is warp divergence. Thirty-two threads execute in lockstep; if they follow different paths through the tree (because they process bodies in different spatial regions), most threads idle while the others work. The Morton sort reduces this significantly: bodies that are spatially adjacent have adjacent Morton indices and therefore adjacent thread indices within a warp, leading to nearly identical traversal paths.

### When GPU is faster than CPU

For the N-squared algorithm, the GPU outperforms the CPU for N >= 1,000, because the kernel is embarrassingly parallel with no tree traversal divergence.

For Barnes-Hut, the GPU advantage is less pronounced for small N due to the fixed overhead of PCIe transfers (~few milliseconds per step regardless of N). The crossover point is typically around N = 5,000 to 10,000 depending on the GPU and CPU models. Above that, the GPU scales better because the O(N log N) force computation dominates and parallelizes well after Morton sorting.
