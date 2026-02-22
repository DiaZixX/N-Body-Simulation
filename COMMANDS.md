# N-Body Simulation - Command Reference

## Graphic Mode (GUI)

### 2D
```bash
# Barnes-Hut (default)
cargo run --release

# Barnes-Hut (explicit)
cargo run --release gui

# N² Direct
cargo run --release gui --direct

# GPU Barnes-Hut
cargo run --release --features cuda

# GPU Barnes-Hut (explicit)
cargo run --release --features cuda gui

# GPU N² Direct
cargo run --release --features cuda gui --direct
```

#### 3D
```bash
# CPU Barnes-Hut (default)
cargo run --release --no-default-features --features vec3

# CPU N² Direct
cargo run --release --no-default-features --features vec3 gui --direct

# GPU Barnes-Hut
cargo run --release --no-default-features --features "vec3,cuda"

# GPU N² Direct
cargo run --release --no-default-features --features "vec3,cuda" gui --direct
```

## Headless Mode (No Interface)

### 2D CPU
```bash
# Barnes-Hut
cargo run --release headless -n 100000

# N² Direct
cargo run --release headless -n 10000 --direct
```

### 2D GPU
```bash
# Barnes-Hut
cargo run --release --features cuda headless -n 500000

# N² Direct
cargo run --release --features cuda headless -n 100000 --direct
```

### 3D CPU
```bash
# Barnes-Hut
cargo run --release --no-default-features --features vec3 headless -n 100000

# N² Direct
cargo run --release --no-default-features --features vec3 headless -n 10000 --direct
```

### 3D GPU
```bash
# Barnes-Hut
cargo run --release --no-default-features --features "vec3,cuda" headless -n 500000

# N² Direct
cargo run --release --no-default-features --features "vec3,cuda" headless -n 100000 --direct
```

## Recap 

### Graphic Mode

| Dimension | Method     | Backend | Command                                                                       |
|-----------|------------|---------|-------------------------------------------------------------------------------|
| **2D**    | Barnes-Hut | CPU     | `cargo run --release`                                                         |
| **2D**    | N²         | CPU     | `cargo run --release gui --direct`                                            |
| **2D**    | Barnes-Hut | CUDA    | `cargo run --release --features cuda`                                         |
| **2D**    | N²         | CUDA    | `cargo run --release --features cuda gui --direct`                            |
| **3D**    | Barnes-Hut | CPU     | `cargo run --release --no-default-features --features vec3`                   |
| **3D**    | N²         | CPU     | `cargo run --release --no-default-features --features vec3 gui --direct`      |
| **3D**    | Barnes-Hut | CUDA    | `cargo run --release --no-default-features --features "vec3,cuda"`            |
| **3D**    | N²         | CUDA    | `cargo run --release --no-default-features --features "vec3,cuda" gui --direct` |

### Mode Headless

| Dimension | Method     | Backend | Command                                                                                     |
|-----------|------------|---------|---------------------------------------------------------------------------------------------|
| **2D**    | Barnes-Hut | CPU     | `cargo run --release headless -n 100000`                                                    |
| **2D**    | N²         | CPU     | `cargo run --release headless -n 10000 --direct`                                            |
| **2D**    | Barnes-Hut | CUDA    | `cargo run --release --features cuda headless -n 500000`                                    |
| **2D**    | N²         | CUDA    | `cargo run --release --features cuda headless -n 100000 --direct`                           |
| **3D**    | Barnes-Hut | CPU     | `cargo run --release --no-default-features --features vec3 headless -n 100000`              |
| **3D**    | N²         | CPU     | `cargo run --release --no-default-features --features vec3 headless -n 10000 --direct`      |
| **3D**    | Barnes-Hut | CUDA    | `cargo run --release --no-default-features --features "vec3,cuda" headless -n 500000`       |
| **3D**    | N²         | CUDA    | `cargo run --release --no-default-features --features "vec3,cuda" headless -n 100000 --direct` |

## Use Examples

### Quick Launch
```bash
# GUI by default (2D CPU Barnes-Hut)
cargo run --release

# GUI 3D GPU Barnes-Hut
cargo run --release --no-default-features --features "vec3,cuda"

# Headless for benchmark
cargo run --release --features cuda headless -n 1000000 -s 100 --no-progress
```

### Performance comparisons
```bash
# Compare CPU vs GPU (2D Barnes-Hut, 100k bodies)
time cargo run --release headless -n 100000 -s 100 --no-progress
time cargo run --release --features cuda headless -n 100000 -s 100 --no-progress

# Compare N² vs Barnes-Hut (2D GPU, 50k bodies)
time cargo run --release --features cuda headless -n 50000 -s 100 --direct --no-progress
time cargo run --release --features cuda headless -n 50000 -s 100 --no-progress

# Compare 2D vs 3D (GPU Barnes-Hut, 100k bodies)
time cargo run --release --features cuda headless -n 100000 -s 100 --no-progress
time cargo run --release --no-default-features --features "vec3,cuda" headless -n 100000 -s 100 --no-progress
```

### Precision test
```bash
# GPU N² (exact reference)
cargo run --release --features cuda headless -n 10000 -s 1000 --direct --dt 0.001 -e 1

# GPU Barnes-Hut high precision
cargo run --release --features cuda headless -n 10000 -s 1000 --theta 0.3 --dt 0.001 -e 1

# CPU N² (reference)
cargo run --release headless -n 10000 -s 1000 --direct --dt 0.001 -e 1
```
