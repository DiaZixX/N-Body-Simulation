//! @file mod.rs
//! @brief Physics body module
//!
//! This module defines the Body structure representing a physical entity
//! in the N-body simulation. Each body has position, velocity, acceleration,
//! mass, and a collision radius.
//!
//! ## Body Properties:
//! - **Position (pos)**: Current location in space
//! - **Velocity (vel)**: Rate of change of position
//! - **Acceleration (acc)**: Rate of change of velocity (from forces)
//! - **Mass**: Gravitational mass of the body
//! - **Radius**: Collision/visual radius
//!
//! ## Integration:
//! Bodies use simple Euler integration for time stepping:
//! - v(t+dt) = v(t) + a(t) * dt
//! - p(t+dt) = p(t) + v(t) * dt
//!
//! ## Dimensionality:
//! The Body structure adapts to 2D or 3D automatically based on the
//! Vector type selected through compilation features.
//!
//! ## Typical Workflow:
//! 1. Create bodies with initial conditions
//! 2. Compute forces → update accelerations
//! 3. Call `update(dt)` to advance simulation
//! 4. Reset accelerations before next force computation

pub mod body;

pub use body::Body;
