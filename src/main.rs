//! @file main.rs
//! @brief Entry point — CLI parsing and subcommand dispatch

use clap::{Parser, Subcommand};
use n_body_simulation::{GuiConfig, SimulationConfig, run, run_headless};

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
        /// Number of bodies to simulate
        #[arg(short = 'n', long, default_value_t = 100_000)]
        num_bodies: usize,

        /// Time step (dt) for integration
        #[arg(long, default_value_t = 0.05)]
        dt: f32,

        /// Barnes-Hut theta parameter (lower = more accurate, slower)
        #[arg(long, default_value_t = 1.0)]
        theta: f32,

        /// Softening parameter epsilon (prevents singularities)
        #[arg(long, default_value_t = 1.0)]
        epsilon: f32,

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

/// @brief Application entry point.
///
/// Parses CLI arguments and dispatches to GUI or headless mode.
/// @return Ok on success, Err on failure
fn main() -> anyhow::Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Gui {
            num_bodies,
            dt,
            theta,
            epsilon,
            direct,
        }) => {
            let config = GuiConfig::new(num_bodies, dt, theta, epsilon, !direct);

            if let Err(e) = config.validate() {
                eprintln!("Configuration error: {}", e);
                std::process::exit(1);
            }

            let method = if direct { "N²" } else { "Barnes-Hut" };

            #[cfg(feature = "cuda")]
            println!("Starting graphical simulation (GPU {})...", method);

            #[cfg(not(feature = "cuda"))]
            println!("Starting graphical simulation (CPU {})...", method);

            println!("Configuration:");
            println!("  Bodies: {}", num_bodies);
            println!("  dt: {}", dt);
            println!("  theta: {}", theta);
            println!("  epsilon: {}", epsilon);
            println!();

            run(config)
        }

        None => {
            // Défaut: GUI avec Barnes-Hut et paramètres par défaut
            let config = GuiConfig::default();

            #[cfg(feature = "cuda")]
            println!("Starting graphical simulation (GPU Barnes-Hut)...");

            #[cfg(not(feature = "cuda"))]
            println!("Starting graphical simulation (CPU Barnes-Hut)...");

            println!("Configuration:");
            println!("  Bodies: {}", config.num_bodies);
            println!("  dt: {}", config.dt);
            println!("  theta: {}", config.theta);
            println!("  epsilon: {}", config.epsilon);
            println!();

            run(config)
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
