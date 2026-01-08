//! @file body.rs
//! @brief Physics body implementation with configurable vector dimensions

use crate::geom::Vector;
use std::fmt;

/// @brief Represents a physical body with position, velocity, acceleration, mass and radius.
///
/// The dimensionality (2D or 3D) is determined at compile-time via Cargo features.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    /// @brief Position of the body
    pub pos: Vector,
    /// @brief Velocity of the body
    pub vel: Vector,
    /// @brief Acceleration of the body
    pub acc: Vector,
    /// @brief Mass of the body
    pub mass: f32,
    /// @brief Radius for collision detection
    pub radius: f32,
}

impl Body {
    /// @brief Creates a new Body with the specified parameters.
    ///
    /// Acceleration is initialized to zero.
    ///
    /// @param pos Initial position
    /// @param vel Initial velocity
    /// @param mass Mass of the body
    /// @param radius Collision radius
    /// @return A new Body instance
    pub fn new(pos: Vector, vel: Vector, mass: f32, radius: f32) -> Self {
        Self {
            pos,
            vel,
            acc: Vector::zero(),
            mass,
            radius,
        }
    }

    /// @brief Updates the body's position and velocity based on acceleration.
    ///
    /// Uses Euler integration: v += a*dt, p += v*dt
    ///
    /// @param dt Time step (delta time)
    pub fn update(&mut self, dt: f32) {
        self.vel += self.acc * dt;
        self.pos += self.vel * dt;
    }

    /// @brief Resets the acceleration to zero.
    ///
    /// This should be called before recomputing forces each frame.
    pub fn reset_acceleration(&mut self) {
        self.acc = Vector::zero();
    }
}

/// @brief Display trait implementation for Body.
///
/// Formats the body as "Body(pos=..., vel=..., mass=..., radius=...)"
/// where position and velocity use the Vector's Display implementation,
/// automatically adapting to 2D or 3D based on the active feature.
/// Mass and radius are displayed with two decimal places.
impl fmt::Display for Body {
    /// @brief Formats the body for display.
    ///
    /// @param f The formatter
    /// @return Result of the formatting operation
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Body(pos={}, vel={}, mass={:.2}, radius={:.2})",
            self.pos, self.vel, self.mass, self.radius
        )
    }
}
