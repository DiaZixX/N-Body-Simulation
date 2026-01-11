//! @file mod.rs
//! @brief Geometric types module
//!
//! This module provides vector mathematics for N-body simulations.
//! It includes implementations for both 2D and 3D vectors with
//! automatic selection based on compilation features.
//!
//! ## Features:
//! - **vec2** (default): 2D vector operations (Vec2)
//! - **vec3**: 3D vector operations (Vec3)
//!
//! ## Vector Operations:
//! - Arithmetic: addition, subtraction, scalar multiplication/division
//! - Geometric: norm, normalization, distance
//! - 3D specific: dot product, cross product (vec3 only)
//!
//! ## Type Alias:
//! The `Vector` type alias automatically resolves to either Vec2 or Vec3
//! based on the active feature, allowing dimension-agnostic code.

pub mod vec2;
pub mod vec3;
pub mod vector;

pub use vector::Vector;
