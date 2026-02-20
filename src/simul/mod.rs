//! @file mod.rs
//! @brief Simulation utilities module
//!
//! This module provides utilities for N-body simulations including:
//! - Body generation functions for various initial configurations
//! - Force computation methods (direct N² and Barnes-Hut)
//! - Initial condition generators (Gaussian, uniform, solar system)
//!
//! The module adapts to 2D (vec2) or 3D (vec3) based on compilation features.
pub mod compute;
pub mod energy;
pub mod generate;
pub mod stats;

pub use compute::compute_nsquares;
pub use energy::{
    kinetic_energy, potential_energy_pair, print_energy_stats, total_kinetic_energy,
    total_mechanical_energy, total_potential_energy,
};
pub use generate::{generate_gaussian, generate_solar_system, generate_uniform, uniform_disc};
pub use stats::PerformanceStats;
