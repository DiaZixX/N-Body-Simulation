//! @file energy.rs
//! @brief Energy calculations for N-body simulations

use crate::body::Body;

/// @brief Gravitational constant (normalized units)
const G: f32 = 1.0;

/// @brief Computes kinetic energy of a body
///
/// E_k = 0.5 * m * v²
///
/// @param body The body
/// @return Kinetic energy
pub fn kinetic_energy(body: &Body) -> f32 {
    0.5 * body.mass * body.vel.norm_squared()
}

/// @brief Computes potential energy between two bodies
///
/// E_p = -G * m1 * m2 / r
///
/// @param body1 First body
/// @param body2 Second body
/// @return Potential energy (negative)
pub fn potential_energy_pair(body1: &Body, body2: &Body) -> f32 {
    let diff = body2.pos - body1.pos;
    let dist = diff.norm();

    if dist < 1e-10 {
        return 0.0;
    }

    -G * body1.mass * body2.mass / dist
}

/// @brief Computes total kinetic energy of the system
///
/// @param bodies Slice of bodies
/// @return Total kinetic energy
pub fn total_kinetic_energy(bodies: &[Body]) -> f32 {
    bodies.iter().map(|b| kinetic_energy(b)).sum()
}

/// @brief Computes total potential energy of the system
///
/// Sums all pairwise potential energies
///
/// @param bodies Slice of bodies
/// @return Total potential energy
pub fn total_potential_energy(bodies: &[Body]) -> f32 {
    let n = bodies.len();
    let mut total = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            total += potential_energy_pair(&bodies[i], &bodies[j]);
        }
    }

    total
}

/// @brief Computes total mechanical energy of the system
///
/// E_total = E_kinetic + E_potential
///
/// @param bodies Slice of bodies
/// @return Total mechanical energy
pub fn total_mechanical_energy(bodies: &[Body]) -> f32 {
    total_kinetic_energy(bodies) + total_potential_energy(bodies)
}

/// @brief Prints energy statistics
///
/// @param bodies Slice of bodies
/// @param step Simulation step number
pub fn print_energy_stats(bodies: &[Body], step: usize) {
    let e_k = total_kinetic_energy(bodies);
    let e_p = total_potential_energy(bodies);
    let e_total = e_k + e_p;

    println!(
        "Step {}: E_kinetic = {:.6e}, E_potential = {:.6e}, E_total = {:.6e}",
        step, e_k, e_p, e_total
    );
}
