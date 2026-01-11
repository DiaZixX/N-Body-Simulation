//! @file main.rs
//! @brief Main entry point for the n-body simulation

use n_body_simulation::{Body, Vector};
use n_body_simulation::{KdCell, KdTree, Node};

/// Constante gravitationnelle (à ajuster selon vos besoins)
const G: f32 = 6.674e-11; // ou une valeur plus adaptée à votre simulation

/// Calcule les accélérations gravitationnelles pour tous les corps
/// Complexité: O(n²)
pub fn compute_nsquares(bodies: &mut [Body]) {
    let n = bodies.len();

    /* for body in &mut *bodies {
        println!(
            "Body BEFORE compute : pos : {} vel : {} acc : {}",
            body.pos, body.vel, body.acc
        );
    } */

    // Réinitialiser les accélérations
    for body in bodies.iter_mut() {
        body.reset_acceleration();
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let delta = bodies[j].pos - bodies[i].pos;
            let distance_sq = delta.x * delta.x + delta.y * delta.y;

            if distance_sq < 1e-10 {
                continue;
            }

            let distance = distance_sq.sqrt();

            let force_direction = delta / distance;

            let accel_i = G * bodies[j].mass / distance_sq;
            let accel_j = G * bodies[i].mass / distance_sq;

            bodies[i].acc += force_direction * accel_i;
            bodies[j].acc -= force_direction * accel_j;
        }
    }

    /* for body in &mut *bodies {
        println!(
            "Body AFTER compute : pos : {} vel : {} acc : {}",
            body.pos, body.vel, body.acc
        );
    } */
}

fn main() {
    println!("=== N-Body Simulation ===\n");

    #[cfg(all(feature = "vec2", not(feature = "vec3")))]
    println!("Running in 2D mode\n");

    #[cfg(feature = "vec3")]
    println!("Running in 3D mode\n");

    // Votre simulation principale ici
    let body = {
        #[cfg(all(feature = "vec2", not(feature = "vec3")))]
        {
            Body::new(Vector::new(0.0, 0.0), Vector::new(1.0, 0.0), 1.0, 0.5)
        }

        #[cfg(feature = "vec3")]
        {
            Body::new(
                Vector::new(0.0, 0.0, 0.0),
                Vector::new(1.0, 0.0, 0.0),
                1.0,
                0.5,
            )
        }
    };

    println!("Created body: {}", body);
    println!("\nSimulation would run here...");
}
