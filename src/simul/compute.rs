//! @file compute.rs
//! @brief Force computation methods for N-body simulations
//!
//! This module provides different algorithms for computing gravitational forces
//! between bodies. Currently implements the direct N² method.

use crate::body::Body;

/// @brief Gravitational constant (m³/kg/s²)
const G: f32 = 6.674e-11;

/// @brief Computes gravitational forces using the direct N² method.
///
/// This is the naive O(N²) algorithm that computes forces between all pairs
/// of bodies directly. It is accurate but computationally expensive for large N.
///
/// @param bodies Mutable slice of bodies whose accelerations will be updated
pub fn compute_nsquares(bodies: &mut [Body]) {
    let n = bodies.len();

    // Step 1: Reset all accelerations
    for body in bodies.iter_mut() {
        body.reset_acceleration();
    }

    // Step 2: Compute pairwise forces
    for i in 0..n {
        for j in (i + 1)..n {
            // Extract data to avoid borrow checker issues
            let pos_i = bodies[i].pos;
            let pos_j = bodies[j].pos;
            let mass_i = bodies[i].mass;
            let mass_j = bodies[j].mass;

            // Compute displacement vector from i to j
            let diff = pos_j - pos_i;
            let dist_sq = diff.norm_squared();

            // Avoid singularity when bodies are too close
            if dist_sq > 1e-10 {
                // Apply Newton's third law: F_ij = -F_ji
                // a = F / m
                let force_direction = diff / diff.norm();

                let accel_i = G * mass_j / dist_sq;
                let accel_j = G * mass_i / dist_sq;

                bodies[i].acc += force_direction * accel_i;
                bodies[j].acc -= force_direction * accel_j;
            }
        }
    }
}
