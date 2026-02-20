//! @file headless.rs
//! @brief Headless simulation runner (no graphics)

use crate::body::Body;
use crate::kdtree::{KdCell, KdTree};
use crate::simul::{PerformanceStats, print_energy_stats, uniform_disc};
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::cuda::compute_forces_cuda;

/// @brief Simulation configuration
pub struct SimulationConfig {
    pub num_bodies: usize,
    pub num_steps: usize,
    pub dt: f32,
    pub theta: f32,
    pub epsilon: f32,
    pub energy_print_interval: usize,
    pub use_barnes_hut: bool,
    pub show_progress: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            num_bodies: 1000,
            num_steps: 1000,
            dt: 0.05,
            theta: 1.0,
            epsilon: 1.0,
            energy_print_interval: 10,
            use_barnes_hut: true,
            show_progress: true,
        }
    }
}

/// @brief Runs a headless simulation
///
/// @param config Simulation configuration
pub fn run_headless(config: SimulationConfig) {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                  N-BODY HEADLESS SIMULATION                   ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Configuration:                                                ║");
    println!(
        "║   Bodies:                       {:>12}                  ║",
        config.num_bodies
    );
    println!(
        "║   Steps:                        {:>12}                  ║",
        config.num_steps
    );
    println!(
        "║   Time step (dt):               {:>12.3}                  ║",
        config.dt
    );
    println!(
        "║   Theta (Barnes-Hut):           {:>12.3}                  ║",
        config.theta
    );
    println!(
        "║   Epsilon (softening):          {:>12.3}                  ║",
        config.epsilon
    );

    #[cfg(feature = "vec2")]
    println!(
        "║   Dimension:                    {:>12}                  ║",
        "2D"
    );

    #[cfg(feature = "vec3")]
    println!(
        "║   Dimension:                    {:>12}                  ║",
        "3D"
    );

    #[cfg(feature = "cuda")]
    println!(
        "║   Mode:                         {:>12}                  ║",
        "CUDA (GPU)"
    );

    #[cfg(not(feature = "cuda"))]
    {
        if config.use_barnes_hut {
            println!(
                "║   Mode:                         {:>12}                  ║",
                "Barnes-Hut"
            );
        } else {
            println!(
                "║   Mode:                         {:>12}                  ║",
                "N² Direct"
            );
        }
    }

    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Initialize performance statistics
    let mut perf_stats = PerformanceStats::new();

    // Generate bodies
    println!("Generating {} bodies...", config.num_bodies);
    let mut bodies = uniform_disc(config.num_bodies);
    println!("Bodies generated.\n");

    #[cfg(not(feature = "cuda"))]
    let mut kdtree = if config.use_barnes_hut {
        Some(KdTree::new(config.theta, config.epsilon))
    } else {
        None
    };

    // Initial energy
    println!("Initial conditions:");
    print_energy_stats(&bodies, 0);
    println!();

    // Simulation loop
    for step in 1..=config.num_steps {
        let step_start = Instant::now();

        // === Force Computation ===
        let force_start = Instant::now();

        #[cfg(feature = "cuda")]
        {
            for body in &mut bodies {
                body.reset_acceleration();
            }
            compute_forces_cuda(&mut bodies, config.epsilon, 1.0);
        }

        #[cfg(not(feature = "cuda"))]
        {
            if let Some(ref mut tree) = kdtree {
                // Barnes-Hut
                let kdcell = KdCell::new_containing(&bodies);
                tree.clear(kdcell);

                for body in &bodies {
                    tree.insert(body.pos, body.mass);
                }

                tree.propagate();

                for body in &mut bodies {
                    body.reset_acceleration();
                    body.acc = tree.acc(body.pos);
                }
            } else {
                // N²
                use crate::simul::compute_nsquares;
                compute_nsquares(&mut bodies);
            }
        }

        let force_time = force_start.elapsed();
        perf_stats.add_force_time(force_time);

        // === Integration ===
        let integration_start = Instant::now();

        for body in &mut bodies {
            body.update(config.dt);
        }

        let integration_time = integration_start.elapsed();
        perf_stats.add_integration_time(integration_time);

        // === Total Step Time ===
        let step_time = step_start.elapsed();
        perf_stats.add_step_time(step_time);

        // === Progress Update ===
        if config.show_progress {
            perf_stats.print_progress(step, config.num_steps);
        }

        // === Energy Statistics ===
        if step % config.energy_print_interval == 0 {
            if config.show_progress {
                println!(); // New line after progress bar
            }
            print_energy_stats(&bodies, step);
        }
    }

    // Final statistics
    if config.show_progress {
        println!(); // New line after last progress update
    }

    println!("\nFinal conditions:");
    print_energy_stats(&bodies, config.num_steps);

    // Print performance summary
    perf_stats.print_summary(config.num_bodies);
}
