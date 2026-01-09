use super::quad::Quad;
use crate::geom::vec2::Vec2;

#[derive(Debug, Clone)]
pub struct Node {
    pub children: usize,
    pub next: usize,
    pub quad: Quad,
    pub pos: Vec2,
    pub mass: f32,
}

impl Node {
    pub fn new(next: usize, quad: Quad) -> Self {
        Self {
            children: 0,
            next,
            quad,
            pos: Vec2::zero(),
            mass: 0.0,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children == 0
    }

    pub fn is_branch(&self) -> bool {
        self.children != 0
    }

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
