//! @file vec2.rs
//! @brief 2D vector implementation with arithmetic operations

#![allow(dead_code)]

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// @brief Simple 2D vector type with basic arithmetic operations.
///
/// This structure represents a two-dimensional vector with floating-point coordinates.
/// It provides common vector operations including normalization, distance calculation,
/// and arithmetic operations through operator overloading.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    /// @brief X coordinate of the vector
    pub x: f32,
    /// @brief Y coordinate of the vector
    pub y: f32,
}

impl Vec2 {
    /// @brief Creates a new Vec2 with the specified coordinates.
    ///
    /// @param x The x-coordinate
    /// @param y The y-coordinate
    /// @return A new Vec2 instance
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// @brief Creates a zero vector (0, 0).
    ///
    /// @return A Vec2 with both coordinates set to 0.0
    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// @brief Computes the vector's norm.
    ///
    /// Calculates the Euclidean norm using the formula: sqrt(x² + y²)
    ///
    /// @return The length of the vector as f32
    pub fn norm(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// @brief Computes the squared norm of the vector.
    ///
    /// This is more efficient than norm() when you only need to compare lengths,
    /// as it avoids the costly square root operation.
    ///
    /// @return The squared length of the vector (x² + y²)
    pub fn norm_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// @brief Returns a normalized version of the vector (norm = 1).
    ///
    /// Creates a unit vector pointing in the same direction as the original vector.
    /// If the vector is zero (norm = 0), returns a zero vector instead to avoid
    /// division by zero.
    ///
    /// @return A normalized Vec2 with magnitude 1, or (0, 0) if the original is zero
    pub fn normalized(&self) -> Self {
        let len = self.norm();
        if len == 0.0 {
            Self::zero()
        } else {
            Self {
                x: self.x / len,
                y: self.y / len,
            }
        }
    }

    /// @brief Calculates the Euclidean distance between two vectors.
    ///
    /// Computes the distance using the formula: |a - b|
    ///
    /// @param a The first vector
    /// @param b The second vector
    /// @return The distance between the two vectors as f32
    pub fn distance(a: Self, b: Self) -> f32 {
        (a - b).norm()
    }
}

/* --- Arithmetic operators --- */

/// @brief Addition operator for Vec2.
///
/// Performs component-wise addition: (x1 + x2, y1 + y2)
impl Add for Vec2 {
    type Output = Self;

    /// @brief Adds two vectors together.
    /// @param other The vector to add
    /// @return A new Vec2 representing the sum
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }
}

/// @brief Addition assignment operator for Vec2.
///
/// Performs in-place component-wise addition.
impl AddAssign for Vec2 {
    /// @brief Adds another vector to this vector in place.
    /// @param other The vector to add
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

/// @brief Subtraction operator for Vec2.
///
/// Performs component-wise subtraction: (x1 - x2, y1 - y2)
impl Sub for Vec2 {
    type Output = Self;

    /// @brief Subtracts one vector from another.
    /// @param other The vector to subtract
    /// @return A new Vec2 representing the difference
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

/// @brief Subtraction assignment operator for Vec2.
///
/// Performs in-place component-wise subtraction.
impl SubAssign for Vec2 {
    /// @brief Subtracts another vector from this vector in place.
    /// @param other The vector to subtract
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

/// @brief Scalar multiplication operator for Vec2.
///
/// Multiplies each component by a scalar value.
impl Mul<f32> for Vec2 {
    type Output = Self;

    /// @brief Multiplies the vector by a scalar.
    /// @param scalar The scalar multiplier
    /// @return A new Vec2 scaled by the scalar
    fn mul(self, scalar: f32) -> Self {
        Self::new(self.x * scalar, self.y * scalar)
    }
}

/// @brief Scalar multiplication assignment operator for Vec2.
///
/// Multiplies each component by a scalar value in place.
impl MulAssign<f32> for Vec2 {
    /// @brief Multiplies this vector by a scalar in place.
    /// @param scalar The scalar multiplier
    fn mul_assign(&mut self, scalar: f32) {
        self.x *= scalar;
        self.y *= scalar;
    }
}

/// @brief Scalar division operator for Vec2.
///
/// Divides each component by a scalar value.
impl Div<f32> for Vec2 {
    type Output = Self;

    /// @brief Divides the vector by a scalar.
    /// @param scalar The scalar divisor
    /// @return A new Vec2 divided by the scalar
    fn div(self, scalar: f32) -> Self {
        Self::new(self.x / scalar, self.y / scalar)
    }
}

/// @brief Scalar division assignment operator for Vec2.
///
/// Divides each component by a scalar value in place.
impl DivAssign<f32> for Vec2 {
    /// @brief Divides this vector by a scalar in place.
    /// @param scalar The scalar divisor
    fn div_assign(&mut self, scalar: f32) {
        self.x /= scalar;
        self.y /= scalar;
    }
}

/* --- Pretty printing (Display) --- */

/// @brief Display trait implementation for Vec2.
///
/// Formats the vector as "(x, y)" with two decimal places.
impl fmt::Display for Vec2 {
    /// @brief Formats the vector for display.
    /// @param f The formatter
    /// @return Result of the formatting operation
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2})", self.x, self.y)
    }
}
