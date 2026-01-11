//! @file mod.rs
//! @brief Simulation utilities module
//!
//! This module provides utilities for N-body simulations including:
//! - Body generation functions for various initial configurations
//! - Force computation methods (direct N² and Barnes-Hut)
//! - Initial condition generators (Gaussian, uniform, solar system)
//!
//! The module adapts to 2D (vec2) or 3D (vec3) based on compilation features.
pub mod generate;

pub use generate::{compute_nsquares, generate_gaussian, generate_solar_system, generate_uniform};
