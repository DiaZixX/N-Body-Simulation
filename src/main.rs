//! @file main.rs
//! @brief Main entry point for the n-body simulation

use n_body_simulation::{Body, Vector};

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
