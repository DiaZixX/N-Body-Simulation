//! @file lib.rs
//! @brief Library entry point for n-body simulation

pub mod body;
pub mod geom;
pub mod headless;
pub mod kdtree;
pub mod renderer;
pub mod simul;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use body::Body;
pub use geom::Vector;
pub use headless::{SimulationConfig, run_headless};
pub use kdtree::{KdCell, KdTree, Node};
pub use renderer::{App, State, run};
pub use simul::{
    PerformanceStats, compute_nsquares, generate_gaussian, generate_solar_system, generate_uniform,
    print_energy_stats, total_mechanical_energy, uniform_disc,
};
