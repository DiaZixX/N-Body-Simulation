//! @file main.rs
//! @brief Main entry point for the n-body simulation

use n_body_simulation::{SimulationConfig, run, run_headless};
use std::env;

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();

    // Check if headless mode
    if args.len() > 1 && args[1] == "--headless" {
        let config = SimulationConfig {
            num_bodies: 10_000,
            num_steps: 100,
            dt: 0.05,
            theta: 1.0,
            epsilon: 1.0,
            energy_print_interval: 10,
            use_barnes_hut: true,
            show_progress: true,
        };

        run_headless(config);
        Ok(())
    } else {
        run()
    }
}
