use super::kdcell::KdCell;
use n_body_simulation::Vector;
use std::fmt;

/// @brief Node in the Barnes-Hut tree structure.
///
/// Represents either a leaf node (containing a single body) or an internal node
/// (containing aggregated mass and center of mass for multiple bodies).
/// The tree structure adapts to 2D (quadtree) or 3D (octree) based on features.
#[derive(Debug, Clone)]
pub struct Node {
    /// @brief Number of children (0 for leaf, 4 for quadtree, 8 for octree)
    pub children: usize,
    /// @brief Index of the next node in the tree array
    pub next: usize,
    /// @brief Spatial cell (k-dimensional bounding box) for this node
    pub kdcell: KdCell,
    /// @brief Position (center of mass for branches, body position for leaves)
    pub pos: Vector,
    /// @brief Total mass (aggregated for branches, body mass for leaves)
    pub mass: f32,
}

impl Node {
    /// @brief Creates a new empty node.
    ///
    /// @param next Index of the next node in the tree
    /// @param kdcell The spatial cell for this node
    /// @return A new Node instance with zero mass
    pub fn new(next: usize, kdcell: KdCell) -> Self {
        Self {
            children: 0,
            next,
            kdcell,
            pos: Vector::zero(),
            mass: 0.0,
        }
    }

    /// @brief Checks if this node is a leaf.
    ///
    /// A leaf node has no children and represents a single body or empty space.
    ///
    /// @return true if the node has no children, false otherwise
    pub fn is_leaf(&self) -> bool {
        self.children == 0
    }

    /// @brief Checks if this node is a branch.
    ///
    /// A branch node has children and represents an aggregated region of space.
    ///
    /// @return true if the node has children, false otherwise
    pub fn is_branch(&self) -> bool {
        self.children != 0
    }

    /// @brief Checks if this node is empty.
    ///
    /// An empty node contains no mass (no bodies in this region).
    ///
    /// @return true if the node has zero mass, false otherwise
    pub fn is_empty(&self) -> bool {
        self.mass == 0.0
    }
}

/// @brief Display trait implementation for Node.
///
/// Formats the node showing its type (Leaf/Branch), position, mass,
/// and spatial cell information. Adapts automatically to 2D or 3D mode.
impl fmt::Display for Node {
    /// @brief Formats the node for display.
    ///
    /// @param f The formatter
    /// @return Result of the formatting operation
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node_type = if self.is_empty() {
            "Empty"
        } else if self.is_leaf() {
            "Leaf"
        } else {
            "Branch"
        };

        write!(
            f,
            "Node[{}](pos={}, mass={:.2}, children={}, cell={})",
            node_type, self.pos, self.mass, self.children, self.kdcell
        )
    }
}
