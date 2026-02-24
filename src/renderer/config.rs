//! @file config.rs
//! @brief Configuration structure for GUI simulations

/// @brief Configuration for the graphical simulation mode.
#[derive(Debug, Clone)]
pub struct GuiConfig {
    pub num_bodies: usize,
    pub dt: f32,
    pub theta: f32,
    pub epsilon: f32,
    pub use_barnes_hut: bool,
}

impl Default for GuiConfig {
    fn default() -> Self {
        Self {
            num_bodies: 100_000,
            dt: 0.05,
            theta: 1.0,
            epsilon: 1.0,
            use_barnes_hut: true,
        }
    }
}

impl GuiConfig {
    /// @brief Creates a new GuiConfig with explicit parameters.
    ///
    /// @param num_bodies Number of bodies to simulate
    /// @param dt Integration time step
    /// @param theta Barnes-Hut opening angle
    /// @param epsilon Softening length
    /// @param use_barnes_hut Whether to use Barnes-Hut (true) or N² (false)
    /// @return New GuiConfig instance
    pub fn new(num_bodies: usize, dt: f32, theta: f32, epsilon: f32, use_barnes_hut: bool) -> Self {
        Self {
            num_bodies,
            dt,
            theta,
            epsilon,
            use_barnes_hut,
        }
    }

    /// @brief Validates the configuration.
    ///
    /// @return Ok if valid, Err with description otherwise
    pub fn validate(&self) -> Result<(), String> {
        if self.num_bodies == 0 {
            return Err("Number of bodies must be greater than 0".to_string());
        }
        if self.dt <= 0.0 {
            return Err("Time step (dt) must be positive".to_string());
        }
        if self.theta <= 0.0 {
            return Err("Theta must be positive".to_string());
        }
        if self.epsilon < 0.0 {
            return Err("Epsilon must be non-negative".to_string());
        }
        Ok(())
    }
}
