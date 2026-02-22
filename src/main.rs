//! @file main.rs
//! @brief Main entry point for the n-body simulation

use clap::{Parser, Subcommand};
use n_body_simulation::{SimulationConfig, run, run_headless};

/// N-Body Simulation: Gravitational dynamics simulator
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run simulation with graphical interface
    Gui {
        /// Use direct N² method instead of Barnes-Hut
        #[arg(long)]
        direct: bool,
    },

    /// Run headless simulation (no graphics)
    Headless {
        /// Number of bodies to simulate
        #[arg(short = 'n', long, default_value_t = 10_000)]
        num_bodies: usize,

        /// Number of simulation steps
        #[arg(short = 's', long, default_value_t = 100)]
        num_steps: usize,

        /// Time step (dt) for integration
        #[arg(long, default_value_t = 0.05)]
        dt: f32,

        /// Barnes-Hut theta parameter (lower = more accurate, slower)
        #[arg(long, default_value_t = 1.0)]
        theta: f32,

        /// Softening parameter epsilon (prevents singularities)
        #[arg(long, default_value_t = 1.0)]
        epsilon: f32,

        /// Interval for printing energy statistics
        #[arg(short = 'e', long, default_value_t = 10)]
        energy_interval: usize,

        /// Use direct N² method instead of Barnes-Hut
        #[arg(long)]
        direct: bool,

        /// Disable progress bar
        #[arg(long)]
        no_progress: bool,
    },
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Gui { direct }) => {
            let method = if direct { "N²" } else { "Barnes-Hut" };

            #[cfg(feature = "cuda")]
            println!("Starting graphical simulation (GPU {})...\n", method);

            #[cfg(not(feature = "cuda"))]
            println!("Starting graphical simulation (CPU {})...\n", method);

            run(!direct) // use_barnes_hut = !direct
        }

        None => {
            // Défaut: GUI avec Barnes-Hut
            #[cfg(feature = "cuda")]
            println!("Starting graphical simulation (GPU Barnes-Hut)...\n");

            #[cfg(not(feature = "cuda"))]
            println!("Starting graphical simulation (CPU Barnes-Hut)...\n");

            run(true)
        }

        Some(Commands::Headless {
            num_bodies,
            num_steps,
            dt,
            theta,
            epsilon,
            energy_interval,
            direct,
            no_progress,
        }) => {
            let config = SimulationConfig {
                num_bodies,
                num_steps,
                dt,
                theta,
                epsilon,
                energy_print_interval: energy_interval,
                use_barnes_hut: !direct,
                show_progress: !no_progress,
            };

            match run_headless(config) {
                Ok(_) => Ok(()),
                Err(e) => {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}
