//! @file quad.rs
//! @brief Spatial partitioning structure for Barnes-Hut algorithm
//!
//! This module provides a spatial cell structure that adapts to the dimensionality:
//! - In 2D mode (vec2): Acts as a Quadtree cell with 4 children
//! - In 3D mode (vec3): Acts as an Octree cell with 8 children

use n_body_simulation::{Body, Vector};
use std::fmt;

/// @brief Spatial cell for hierarchical space partitioning.
///
/// This structure represents a cubic (3D) or square (2D) region of space.
/// It is used in the Barnes-Hut algorithm for efficient N-body simulation.
/// The cell adapts its behavior based on the active feature flag:
/// - vec2: Quadtree cell (4 subdivisions)
/// - vec3: Octree cell (8 subdivisions)
#[derive(Clone, Copy, Debug)]
pub struct KdCell {
    /// @brief Center point of the spatial cell
    pub center: Vector,
    /// @brief Size of the cell (side length of the square/cube)
    pub size: f32,
}

impl KdCell {
    /// @brief Creates a new spatial cell.
    ///
    /// @param center The center point of the cell
    /// @param size The side length of the cell
    /// @return A new Quad instance
    pub fn new(center: Vector, size: f32) -> Self {
        Self { center, size }
    }

    /// @brief Subdivides into a specific quadrant (2D mode only).
    ///
    /// Creates a child quadrant by halving the size and offsetting the center.
    /// Quadrant indexing:
    /// - 0: Bottom-left
    /// - 1: Bottom-right
    /// - 2: Top-left
    /// - 3: Top-right
    ///
    /// @param i The quadrant index (0-3)
    /// @return A new Quad representing the specified quadrant
    #[cfg(feature = "vec2")]
    pub fn into_quadrant(mut self, i: usize) -> Self {
        self.size *= 0.5;
        self.center.x += (0.5 - (i & 1) as f32) * self.size;
        self.center.y += (0.5 - (i >> 1) as f32) * self.size;
        self
    }

    /// @brief Subdivides into a specific octant (3D mode only).
    ///
    /// Creates a child octant by halving the size and offsetting the center.
    /// Octant indexing uses 3 bits (x, y, z):
    /// - Bit 0: x direction (0=left, 1=right)
    /// - Bit 1: y direction (0=bottom, 1=top)
    /// - Bit 2: z direction (0=back, 1=front)
    ///
    /// @param i The octant index (0-7)
    /// @return A new Quad representing the specified octant
    #[cfg(feature = "vec3")]
    pub fn into_octant(mut self, i: usize) -> Self {
        self.size *= 0.5;
        self.center.x += (0.5 - (i & 1) as f32) * self.size;
        self.center.y += (0.5 - ((i >> 1) & 1) as f32) * self.size;
        self.center.z += (0.5 - (i >> 2) as f32) * self.size;
        self
    }

    /// @brief Generates all 4 quadrant subdivisions (2D mode only).
    ///
    /// Creates an array of all possible quadrant children for this cell.
    ///
    /// @return Array of 4 Quad instances representing all quadrants
    #[cfg(feature = "vec2")]
    pub fn into_quadrants(&self) -> [KdCell; 4] {
        [0, 1, 2, 3].map(|i| self.into_quadrant(i))
    }

    /// @brief Generates all 8 octant subdivisions (3D mode only).
    ///
    /// Creates an array of all possible octant children for this cell.
    ///
    /// @return Array of 8 Quad instances representing all octants
    #[cfg(feature = "vec3")]
    pub fn into_octants(&self) -> [KdCell; 8] {
        [0, 1, 2, 3, 4, 5, 6, 7].map(|i| self.into_octant(i))
    }

    /// @brief Finds which quadrant a position belongs to (2D mode only).
    ///
    /// Determines the quadrant index based on position relative to center.
    /// Uses bit manipulation for efficient calculation:
    /// - Bit 0: 1 if x > center.x (right), 0 otherwise (left)
    /// - Bit 1: 1 if y > center.y (top), 0 otherwise (bottom)
    ///
    /// @param pos The position to locate
    /// @return The quadrant index (0-3)
    #[cfg(feature = "vec2")]
    pub fn find_quadrant(&self, pos: Vector) -> usize {
        ((pos.y > self.center.y) as usize) << 1 | (pos.x > self.center.x) as usize
    }

    /// @brief Finds which octant a position belongs to (3D mode only).
    ///
    /// Determines the octant index based on position relative to center.
    /// Uses bit manipulation for efficient calculation:
    /// - Bit 0: 1 if x > center.x (right), 0 otherwise (left)
    /// - Bit 1: 1 if y > center.y (top), 0 otherwise (bottom)
    /// - Bit 2: 1 if z > center.z (front), 0 otherwise (back)
    ///
    /// @param pos The position to locate
    /// @return The octant index (0-7)
    #[cfg(feature = "vec3")]
    pub fn find_octant(&self, pos: Vector) -> usize {
        ((pos.z > self.center.z) as usize) << 2
            | ((pos.y > self.center.y) as usize) << 1
            | (pos.x > self.center.x) as usize
    }

    /// @brief Creates a spatial cell containing all given bodies.
    ///
    /// Computes the bounding box that encompasses all body positions,
    /// then creates a cubic/square cell centered on this region.
    /// The size is chosen as the maximum extent in any dimension.
    ///
    /// In 2D mode: Considers only x and y coordinates
    /// In 3D mode: Considers x, y, and z coordinates
    ///
    /// @param bodies Slice of bodies to contain
    /// @return A new Quad that contains all body positions
    pub fn new_containing(bodies: &[Body]) -> Self {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        #[cfg(feature = "vec3")]
        let mut min_z = f32::MAX;
        #[cfg(feature = "vec3")]
        let mut max_z = f32::MIN;

        for body in bodies {
            min_x = min_x.min(body.pos.x);
            min_y = min_y.min(body.pos.y);
            max_x = max_x.max(body.pos.x);
            max_y = max_y.max(body.pos.y);

            #[cfg(feature = "vec3")]
            {
                min_z = min_z.min(body.pos.z);
                max_z = max_z.max(body.pos.z);
            }
        }

        let center = {
            #[cfg(feature = "vec2")]
            {
                Vector::new(min_x + max_x, min_y + max_y) * 0.5
            }

            #[cfg(feature = "vec3")]
            {
                Vector::new(min_x + max_x, min_y + max_y, min_z + max_z) * 0.5
            }
        };

        let size = {
            #[cfg(feature = "vec2")]
            {
                (max_x - min_x).max(max_y - min_y)
            }

            #[cfg(feature = "vec3")]
            {
                (max_x - min_x).max(max_y - min_y).max(max_z - min_z)
            }
        };

        Self { center, size }
    }

    /// @brief Checks if a position is contained within this cell.
    ///
    /// A position is considered inside if all coordinates are within
    /// size/2 distance from the center.
    ///
    /// In 2D mode: Checks x and y coordinates
    /// In 3D mode: Checks x, y, and z coordinates
    ///
    /// @param pos The position to test
    /// @return true if the position is inside the cell, false otherwise
    pub fn contains(&self, pos: Vector) -> bool {
        let diff = self.center - pos;

        #[cfg(feature = "vec2")]
        {
            diff.x.abs() <= self.size / 2.0 && diff.y.abs() <= self.size / 2.0
        }

        #[cfg(feature = "vec3")]
        {
            diff.x.abs() <= self.size / 2.0
                && diff.y.abs() <= self.size / 2.0
                && diff.z.abs() <= self.size / 2.0
        }
    }
}

/// @brief Display trait implementation for KdCell.
///
/// Formats the body as "KdCell(center=..., size=...)"
/// where center is the center of the cell (2D or 3D)
/// and the size is the cell's size.
impl fmt::Display for KdCell {
    /// @brief Formats the spatial cell for display.
    ///
    /// Displays the cell's center position and size.
    /// The format automatically adapts to show 2D or 3D coordinates
    /// based on the Vector's Display implementation.
    ///
    /// @param f The formatter
    /// @return Result of the formatting operation
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KdCell(center={}, size={:.2})", self.center, self.size)
    }
}
