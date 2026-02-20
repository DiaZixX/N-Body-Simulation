//! @file stats.rs
//! @brief Performance statistics for simulations

use std::time::{Duration, Instant};

/// @brief Performance statistics collector
pub struct PerformanceStats {
    pub step_times: Vec<Duration>,
    pub force_computation_times: Vec<Duration>,
    pub integration_times: Vec<Duration>,
    pub total_start: Instant,
}

impl PerformanceStats {
    /// @brief Creates a new performance statistics collector
    pub fn new() -> Self {
        Self {
            step_times: Vec::new(),
            force_computation_times: Vec::new(),
            integration_times: Vec::new(),
            total_start: Instant::now(),
        }
    }

    /// @brief Adds a step timing measurement
    pub fn add_step_time(&mut self, duration: Duration) {
        self.step_times.push(duration);
    }

    /// @brief Adds a force computation timing measurement
    pub fn add_force_time(&mut self, duration: Duration) {
        self.force_computation_times.push(duration);
    }

    /// @brief Adds an integration timing measurement
    pub fn add_integration_time(&mut self, duration: Duration) {
        self.integration_times.push(duration);
    }

    /// @brief Computes mean duration from a slice
    fn mean_duration(durations: &[Duration]) -> Duration {
        if durations.is_empty() {
            return Duration::ZERO;
        }
        let sum: Duration = durations.iter().sum();
        sum / durations.len() as u32
    }

    /// @brief Computes standard deviation of durations
    fn std_dev_duration(durations: &[Duration]) -> f64 {
        if durations.len() < 2 {
            return 0.0;
        }

        let mean = Self::mean_duration(durations).as_secs_f64();
        let variance: f64 = durations
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - mean;
                diff * diff
            })
            .sum::<f64>()
            / durations.len() as f64;

        variance.sqrt()
    }

    /// @brief Finds minimum duration
    fn min_duration(durations: &[Duration]) -> Duration {
        durations.iter().copied().min().unwrap_or(Duration::ZERO)
    }

    /// @brief Finds maximum duration
    fn max_duration(durations: &[Duration]) -> Duration {
        durations.iter().copied().max().unwrap_or(Duration::ZERO)
    }

    /// @brief Computes median duration
    fn median_duration(durations: &[Duration]) -> Duration {
        if durations.is_empty() {
            return Duration::ZERO;
        }

        let mut sorted = durations.to_vec();
        sorted.sort();

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2
        } else {
            sorted[mid]
        }
    }

    /// @brief Computes percentile (0-100)
    fn percentile_duration(durations: &[Duration], percentile: f64) -> Duration {
        if durations.is_empty() {
            return Duration::ZERO;
        }

        let mut sorted = durations.to_vec();
        sorted.sort();

        let index = ((percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    /// @brief Prints detailed performance statistics
    pub fn print_summary(&self, num_bodies: usize) {
        let total_time = self.total_start.elapsed();

        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║           PERFORMANCE STATISTICS SUMMARY                      ║");
        println!("╠═══════════════════════════════════════════════════════════════╣");

        // Overall statistics
        println!("║ Overall Statistics:                                           ║");
        println!(
            "║   Total simulation time:        {:>12.3} s                ║",
            total_time.as_secs_f64()
        );
        println!(
            "║   Number of bodies:             {:>12}                  ║",
            num_bodies
        );
        println!(
            "║   Total steps completed:        {:>12}                  ║",
            self.step_times.len()
        );

        if !self.step_times.is_empty() {
            let steps_per_sec = self.step_times.len() as f64 / total_time.as_secs_f64();
            println!(
                "║   Steps per second:             {:>12.2}                  ║",
                steps_per_sec
            );

            let interactions_per_step = (num_bodies * (num_bodies - 1)) / 2;
            let total_interactions = interactions_per_step * self.step_times.len();
            let interactions_per_sec = total_interactions as f64 / total_time.as_secs_f64();

            println!(
                "║   Interactions per step:        {:>12.2e}                  ║",
                interactions_per_step as f64
            );
            println!(
                "║   Interactions per second:      {:>12.2e}                  ║",
                interactions_per_sec
            );
        }

        println!("╠═══════════════════════════════════════════════════════════════╣");

        // Step timing statistics
        if !self.step_times.is_empty() {
            println!("║ Per-Step Timing:                                              ║");
            self.print_timing_stats("Total step time", &self.step_times);
        }

        // Force computation statistics
        if !self.force_computation_times.is_empty() {
            println!("╠═══════════════════════════════════════════════════════════════╣");
            println!("║ Force Computation Timing:                                     ║");
            self.print_timing_stats("Force computation", &self.force_computation_times);

            // Percentage of step time
            let mean_step = Self::mean_duration(&self.step_times).as_secs_f64();
            let mean_force = Self::mean_duration(&self.force_computation_times).as_secs_f64();
            let force_percentage = if mean_step > 0.0 {
                (mean_force / mean_step) * 100.0
            } else {
                0.0
            };
            println!(
                "║   Percentage of step time:      {:>12.1} %                ║",
                force_percentage
            );
        }

        // Integration statistics
        if !self.integration_times.is_empty() {
            println!("╠═══════════════════════════════════════════════════════════════╣");
            println!("║ Integration Timing:                                           ║");
            self.print_timing_stats("Integration", &self.integration_times);

            // Percentage of step time
            let mean_step = Self::mean_duration(&self.step_times).as_secs_f64();
            let mean_integration = Self::mean_duration(&self.integration_times).as_secs_f64();
            let integration_percentage = if mean_step > 0.0 {
                (mean_integration / mean_step) * 100.0
            } else {
                0.0
            };
            println!(
                "║   Percentage of step time:      {:>12.1} %                ║",
                integration_percentage
            );
        }

        println!("╚═══════════════════════════════════════════════════════════════╝");
    }

    /// @brief Prints timing statistics for a specific category
    fn print_timing_stats(&self, category: &str, durations: &[Duration]) {
        let mean = Self::mean_duration(durations);
        let std_dev = Self::std_dev_duration(durations);
        let min = Self::min_duration(durations);
        let max = Self::max_duration(durations);
        let median = Self::median_duration(durations);
        let p95 = Self::percentile_duration(durations, 95.0);
        let p99 = Self::percentile_duration(durations, 99.0);

        println!(
            "║   Mean:                         {:>12.6} s                ║",
            mean.as_secs_f64()
        );
        println!(
            "║   Std deviation:                {:>12.6} s                ║",
            std_dev
        );
        println!(
            "║   Median:                       {:>12.6} s                ║",
            median.as_secs_f64()
        );
        println!(
            "║   Min:                          {:>12.6} s                ║",
            min.as_secs_f64()
        );
        println!(
            "║   Max:                          {:>12.6} s                ║",
            max.as_secs_f64()
        );
        println!(
            "║   95th percentile:              {:>12.6} s                ║",
            p95.as_secs_f64()
        );
        println!(
            "║   99th percentile:              {:>12.6} s                ║",
            p99.as_secs_f64()
        );
    }

    /// @brief Pads a string to a specific length
    fn pad_string(s: &str, length: usize) -> String {
        if s.len() >= length {
            s[..length].to_string()
        } else {
            format!("{}{}", s, " ".repeat(length - s.len()))
        }
    }

    /// @brief Prints a progress bar for live updates
    pub fn print_progress(&self, current_step: usize, total_steps: usize) {
        let percentage = (current_step as f64 / total_steps as f64) * 100.0;
        let bar_length = 50;
        let filled = ((percentage / 100.0) * bar_length as f64) as usize;
        let empty = bar_length - filled;

        let bar = format!("[{}{}]", "█".repeat(filled), "░".repeat(empty));

        let mean_step_time = if !self.step_times.is_empty() {
            Self::mean_duration(&self.step_times).as_secs_f64()
        } else {
            0.0
        };

        let remaining_steps = total_steps - current_step;
        let eta = mean_step_time * remaining_steps as f64;

        print!(
            "\r{} {:.1}% | Step {}/{} | Avg: {:.3}s/step | ETA: {:.1}s",
            bar, percentage, current_step, total_steps, mean_step_time, eta
        );

        use std::io::{self, Write};
        io::stdout().flush().unwrap();

        if current_step == total_steps {
            println!(); // New line at the end
        }
    }
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self::new()
    }
}
