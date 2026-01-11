//! @file generate.rs
//! @brief Body generation utilities for N-body simulations
//!
//! This module provides functions to generate various initial configurations
//! of bodies for simulation, including Gaussian distributions and solar systems.
//! The generators adapt to 2D (vec2) or 3D (vec3) based on features.

use crate::body::Body;
use crate::geom::Vector;
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

/// @brief Generates bodies in a Gaussian distribution.
///
/// Creates N bodies distributed according to a normal distribution around a center point.
/// Bodies are given orbital velocities perpendicular to their radial direction.
///
/// In 2D mode: Distribution in x-y plane
/// In 3D mode: Distribution in x-y-z space with rotation in x-y plane
///
/// @param n Number of bodies to generate
/// @param center Center point of the distribution
/// @param sigma Standard deviation of the Gaussian distribution
/// @param mass Mass of each body
/// @param radius Collision radius of each body
/// @return Vector of generated bodies
pub fn generate_gaussian(
    n: usize,
    center: Vector,
    sigma: f32,
    mass: f32,
    radius: f32,
) -> Vec<Body> {
    let normal_x = Normal::new(center.x, sigma).unwrap();
    let normal_y = Normal::new(center.y, sigma).unwrap();

    #[cfg(feature = "vec3")]
    let normal_z = Normal::new(center.z, sigma).unwrap();

    let mut rng = rand::thread_rng();
    let mut bodies = Vec::with_capacity(n);

    for _ in 0..n {
        let pos = {
            #[cfg(feature = "vec2")]
            {
                let mut p = Vector::new(normal_x.sample(&mut rng), normal_y.sample(&mut rng));
                // Clamp position
                p.x = p.x.clamp(-10.0, 10.0);
                p.y = p.y.clamp(-10.0, 10.0);
                p
            }

            #[cfg(feature = "vec3")]
            {
                let mut p = Vector::new(
                    normal_x.sample(&mut rng),
                    normal_y.sample(&mut rng),
                    normal_z.sample(&mut rng),
                );
                // Clamp position
                p.x = p.x.clamp(-10.0, 10.0);
                p.y = p.y.clamp(-10.0, 10.0);
                p.z = p.z.clamp(-10.0, 10.0);
                p
            }
        };

        // Calculate vector from center
        let delta = pos - center;
        let distance = delta.norm();

        // Circular orbital velocity: v = sqrt(G * M_central / r)
        let orbital_speed = if distance > 1e-6 {
            // Simplified formula: velocity proportional to 1/sqrt(distance)
            let speed_factor = 0.5;
            speed_factor / distance.sqrt()
        } else {
            0.0
        };

        // Perpendicular vector for rotation (counterclockwise in x-y plane)
        let vel = {
            #[cfg(feature = "vec2")]
            {
                Vector::new(-delta.y, delta.x).normalized() * orbital_speed
            }

            #[cfg(feature = "vec3")]
            {
                // Rotation in x-y plane, z component stays zero
                Vector::new(-delta.y, delta.x, 0.0).normalized() * orbital_speed
            }
        };

        bodies.push(Body::new(pos, vel, mass, radius));
    }
    bodies
}

/// @brief Generates a simplified solar system.
///
/// Creates a mini solar system with the Sun at the center and planets
/// in circular orbits. Uses realistic masses and distances, with velocities
/// calculated using Kepler's laws.
///
/// In 2D mode: Planets orbit in the x-y plane
/// In 3D mode: Planets orbit in the x-y plane (z=0 for simplicity)
///
/// @return Vector of bodies (Sun + planets)
pub fn generate_solar_system() -> Vec<Body> {
    // Gravitational constant (m³/kg/s²)
    const G: f32 = 6.674e-11;

    // System center (Sun will be at center)
    let center = {
        #[cfg(feature = "vec2")]
        {
            Vector::new(0.0, 0.0)
        }

        #[cfg(feature = "vec3")]
        {
            Vector::new(0.0, 0.0, 0.0)
        }
    };

    // Sun's mass for orbital velocity calculations
    let m_sun = 1.989e30;

    let mut bodies = Vec::new();

    // ========== SUN ==========
    // Place it at center with zero velocity
    bodies.push(Body::new(
        center,
        Vector::zero(),
        m_sun, // real mass
        0.010, // visual radius (enlarged to be visible)
    ));

    // ========== PLANETS ==========
    // Format: (name, distance_from_sun_in_m, mass_in_kg, visual_radius)
    let planets = [
        // Mercury
        ("Mercury", 57.9e9, 3.301e23, 0.0016),
        // Venus
        ("Venus", 108.2e9, 4.867e24, 0.0024),
        // Earth
        ("Earth", 149.6e9, 5.972e24, 0.0026),
        // Mars
        ("Mars", 227.9e9, 6.417e23, 0.0018),
        // Jupiter (gas giant)
        ("Jupiter", 778.5e9, 1.898e27, 0.0070),
        // Saturn (gas giant)
        ("Saturn", 1.434e12, 5.683e26, 0.0060),
        // Uranus (ice giant)
        ("Uranus", 2.871e12, 8.681e25, 0.0040),
        // Neptune (ice giant)
        ("Neptune", 4.495e12, 1.024e26, 0.0038),
    ];

    for (i, &(_name, distance, mass, visual_radius)) in planets.iter().enumerate() {
        // Planet position
        // Place each planet at a different angle to avoid initial collisions
        let angle = i as f32 * 2.0 * std::f32::consts::PI / planets.len() as f32;
        let scaled_distance = distance;

        let pos = {
            #[cfg(feature = "vec2")]
            {
                Vector::new(
                    center.x + scaled_distance * angle.cos(),
                    center.y + scaled_distance * angle.sin(),
                )
            }

            #[cfg(feature = "vec3")]
            {
                Vector::new(
                    center.x + scaled_distance * angle.cos(),
                    center.y + scaled_distance * angle.sin(),
                    0.0, // All planets in the same plane for simplicity
                )
            }
        };

        // Calculate circular orbital velocity
        // Formula: v = sqrt(G * M_sun / r)
        let orbital_velocity = (G * m_sun / scaled_distance).sqrt();

        // Velocity vector perpendicular to radius (circular motion)
        // For counterclockwise motion, use (-sin, cos) instead of (cos, sin)
        let vel = {
            #[cfg(feature = "vec2")]
            {
                Vector::new(
                    -orbital_velocity * angle.sin(),
                    orbital_velocity * angle.cos(),
                )
            }

            #[cfg(feature = "vec3")]
            {
                Vector::new(
                    -orbital_velocity * angle.sin(),
                    orbital_velocity * angle.cos(),
                    0.0, // No velocity in z direction
                )
            }
        };

        bodies.push(Body::new(pos, vel, mass, visual_radius));
    }

    println!("=== Mini Solar System Generated ===");
    println!("Sun's mass: {}", m_sun);
    println!("Gravitational constant G: {}", G);
    for (i, body) in bodies.iter().enumerate().skip(1) {
        let r = body.pos.norm();
        let v = body.vel.norm();
        let v_theoretical = (G * m_sun / r).sqrt();
        println!(
            "Planet {}: r={:.3e} m, v={:.6e} m/s, v_theoretical={:.6e} m/s",
            i, r, v, v_theoretical
        );
    }

    bodies
}

/// @brief Generates a uniform random distribution of bodies.
///
/// Creates N bodies uniformly distributed in a cube/square region.
/// Bodies are given random velocities.
///
/// @param n Number of bodies to generate
/// @param min_pos Minimum corner of the bounding box
/// @param max_pos Maximum corner of the bounding box
/// @param max_vel Maximum velocity magnitude
/// @param mass Mass of each body
/// @param radius Collision radius of each body
/// @return Vector of generated bodies
pub fn generate_uniform(
    n: usize,
    min_pos: Vector,
    max_pos: Vector,
    max_vel: f32,
    mass: f32,
    radius: f32,
) -> Vec<Body> {
    let mut rng = rand::thread_rng();
    let mut bodies = Vec::with_capacity(n);

    let uniform_x = Uniform::new(min_pos.x, max_pos.x);
    let uniform_y = Uniform::new(min_pos.y, max_pos.y);
    let uniform_vel = Uniform::new(-max_vel, max_vel);

    #[cfg(feature = "vec3")]
    let uniform_z = Uniform::new(min_pos.z, max_pos.z);

    for _ in 0..n {
        let pos = {
            #[cfg(feature = "vec2")]
            {
                Vector::new(uniform_x.sample(&mut rng), uniform_y.sample(&mut rng))
            }

            #[cfg(feature = "vec3")]
            {
                Vector::new(
                    uniform_x.sample(&mut rng),
                    uniform_y.sample(&mut rng),
                    uniform_z.sample(&mut rng),
                )
            }
        };

        let vel = {
            #[cfg(feature = "vec2")]
            {
                Vector::new(uniform_vel.sample(&mut rng), uniform_vel.sample(&mut rng))
            }

            #[cfg(feature = "vec3")]
            {
                Vector::new(
                    uniform_vel.sample(&mut rng),
                    uniform_vel.sample(&mut rng),
                    uniform_vel.sample(&mut rng),
                )
            }
        };

        bodies.push(Body::new(pos, vel, mass, radius));
    }

    bodies
}
