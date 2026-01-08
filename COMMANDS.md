# N-Body Simulation - Command Reference

## Project Structure
```
n-body-simulation/
├── Cargo.toml
├── src/
│   ├── main.rs          # Main simulation entry point
│   ├── lib.rs           # Library entry point
│   ├── body/            # Physics body implementation
│   │   ├── mod.rs
│   │   └── body.rs
│   └── geom/            # Geometric types (vectors)
│       ├── mod.rs
│       ├── vec2.rs      # 2D vector implementation
│       ├── vec3.rs      # 3D vector implementation
│       └── vector.rs    # Conditional vector type
└── tests/
    └── vector_tests.rs  # Integration tests
```

## Build Commands

### Compilation
```bash
# Build the project (2D mode, default)
cargo build

# Build the project (3D mode)
cargo build --no-default-features --features vec3

# Build with optimizations (release mode)
cargo build --release

# Build 3D with optimizations
cargo build --release --no-default-features --features vec3
```

### Check (faster than build, no executable)
```bash
# Check compilation without building
cargo check

# Check 3D mode
cargo check --no-default-features --features vec3
```

## Run Commands

### Running the Simulation
```bash
# Run the simulation in 2D mode (default)
cargo run

# Run the simulation in 3D mode
cargo run --no-default-features --features vec3

# Run with release optimizations (2D)
cargo run --release

# Run with release optimizations (3D)
cargo run --release --no-default-features --features vec3
```

## Test Commands

### Running All Tests
```bash
# Run all tests in 2D mode
cargo test

# Run all tests in 3D mode
cargo test --no-default-features --features vec3

# Run tests with verbose output
cargo test -- --nocapture

# Run tests and show all output (including successful tests)
cargo test -- --show-output
```

### Running Specific Tests
```bash
# Run a specific test by name
cargo test test_vector_addition

# Run all tests containing "vector" in the name
cargo test vector

# Run all tests containing "body" in the name
cargo test body

# Run a specific test in 3D mode
cargo test test_vector_addition --no-default-features --features vec3
```

### Test with Coverage (requires additional setup)
```bash
# Run tests and show which code is covered
cargo test --verbose
```

## Clean Commands
```bash
# Remove build artifacts
cargo clean

# Clean and rebuild
cargo clean && cargo build
```

## Documentation Commands
```bash
# Generate and open documentation
cargo doc --open

# Generate documentation without dependencies
cargo doc --no-deps --open

# Generate documentation for 3D mode
cargo doc --no-default-features --features vec3 --open
```

## Utility Commands
```bash
# Format code according to Rust style guidelines
cargo fmt

# Check formatting without modifying files
cargo fmt -- --check

# Lint code with Clippy
cargo clippy

# Lint with all warnings
cargo clippy -- -W clippy::all
```

## Feature Flags

The project supports two mutually exclusive features:

- `vec2` (default): 2D vector operations
- `vec3`: 3D vector operations with additional dot and cross product

### Using Features
```bash
# Explicitly enable vec2 (same as default)
cargo run --features vec2

# Disable default features and enable vec3
cargo run --no-default-features --features vec3

# Test with vec3
cargo test --no-default-features --features vec3
```

## Quick Reference Table

| Command | Description |
|---------|-------------|
| `cargo build` | Compile in 2D mode |
| `cargo build --features vec3` | Compile in 3D mode |
| `cargo run` | Run simulation (2D) |
| `cargo run --features vec3` | Run simulation (3D) |
| `cargo test` | Run all tests (2D) |
| `cargo test --features vec3` | Run all tests (3D) |
| `cargo test test_name` | Run specific test |
| `cargo clean` | Remove build artifacts |
| `cargo doc --open` | Generate and view documentation |
| `cargo fmt` | Format code |
| `cargo clippy` | Lint code |

## Common Workflows

### Development Workflow
```bash
# 1. Make changes to code
# 2. Check if it compiles
cargo check

# 3. Run tests
cargo test

# 4. Format code
cargo fmt

# 5. Lint code
cargo clippy

# 6. Run the application
cargo run
```

### Testing Both Modes
```bash
# Test 2D
cargo test

# Test 3D
cargo test --no-default-features --features vec3
```

### Release Build
```bash
# Build optimized binary
cargo build --release

# The binary will be in: target/release/n-body-simulation

# Run optimized version
./target/release/n-body-simulation
```

## Environment Variables
```bash
# Show more detailed error messages
RUST_BACKTRACE=1 cargo run

# Full backtrace
RUST_BACKTRACE=full cargo run

# Enable logging (if using a logger)
RUST_LOG=debug cargo run
```

## Tips

- Use `cargo run` for quick iterations during development
- Use `cargo build --release` for production builds (much faster execution)
- Use `cargo test` frequently to catch bugs early
- Use `cargo clippy` to find common mistakes and improvements
- The `--no-default-features --features vec3` can be shortened in a `.cargo/config.toml` if you work primarily in 3D mode

## Troubleshooting

### Clear build cache if you encounter weird errors
```bash
cargo clean
cargo build
```

### Check which features are enabled
```bash
cargo build --verbose
```

### Update dependencies
```bash
cargo update
```
