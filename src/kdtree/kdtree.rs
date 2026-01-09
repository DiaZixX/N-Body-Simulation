//! @file kdtree.rs
//! @brief K-dimensional tree implementation for Barnes-Hut N-body simulation
//!
//! This module implements a spatial hierarchical tree structure that adapts
//! to the dimensionality:
//! - 2D mode (vec2): Quadtree with 4 children per node
//! - 3D mode (vec3): Octree with 8 children per node
//!
//! The tree is used in the Barnes-Hut algorithm to efficiently compute
//! gravitational forces between bodies by approximating distant groups
//! of bodies as single masses.

use super::{KdCell, Node};
use n_body_simulation::{Body, Vector};
use std::fmt;

/// @brief K-dimensional tree for spatial partitioning.
///
/// Implements a hierarchical spatial data structure:
/// - Quadtree in 2D mode (4 children per branch)
/// - Octree in 3D mode (8 children per branch)
///
/// The tree stores bodies and aggregates their mass and center of mass
/// for efficient force computation using the Barnes-Hut algorithm.
#[derive(Debug)]
pub struct KdTree {
    /// @brief Flat array of all nodes in the tree
    pub nodes: Vec<Node>,
    /// @brief Stack of parent node indices for propagation
    pub parents: Vec<usize>,
}

impl KdTree {
    /// @brief Root node index (always 0)
    const ROOT: usize = 0;

    /// @brief Number of children per node based on dimensionality
    #[cfg(feature = "vec2")]
    const NUM_CHILDREN: usize = 4;

    #[cfg(feature = "vec3")]
    const NUM_CHILDREN: usize = 8;

    /// @brief Creates a new empty KdTree.
    ///
    /// @return A new KdTree instance with no nodes
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            parents: Vec::new(),
        }
    }

    /// @brief Clears the tree and initializes with a root cell.
    ///
    /// @param kdcell The bounding cell for the root node
    pub fn clear(&mut self, kdcell: KdCell) {
        self.nodes.clear();
        self.parents.clear();
        self.nodes.push(Node::new(0, kdcell));
    }

    /// @brief Subdivides a node into children.
    ///
    /// Creates 4 children (quadtree) or 8 children (octree) for the given node.
    ///
    /// @param node Index of the node to subdivide
    /// @return Index of the first child
    fn subdivide(&mut self, node: usize) -> usize {
        self.parents.push(node);
        let children = self.nodes.len();
        self.nodes[node].children = children;

        #[cfg(feature = "vec2")]
        {
            let nexts = [
                children + 1,
                children + 2,
                children + 3,
                self.nodes[node].next,
            ];
            let cells = self.nodes[node].kdcell.into_quadrants();
            for i in 0..4 {
                self.nodes.push(Node::new(nexts[i], cells[i]));
            }
        }

        #[cfg(feature = "vec3")]
        {
            let nexts = [
                children + 1,
                children + 2,
                children + 3,
                children + 4,
                children + 5,
                children + 6,
                children + 7,
                self.nodes[node].next,
            ];
            let cells = self.nodes[node].kdcell.into_octants();
            for i in 0..8 {
                self.nodes.push(Node::new(nexts[i], cells[i]));
            }
        }

        children
    }

    /// @brief Inserts a body into the tree.
    ///
    /// Traverses the tree to find the appropriate leaf node for the body.
    /// If a leaf already contains a body, subdivides until bodies are separated.
    ///
    /// @param pos Position of the body
    /// @param mass Mass of the body
    pub fn insert(&mut self, pos: Vector, mass: f32) {
        let mut node = Self::ROOT;

        while self.nodes[node].is_branch() {
            #[cfg(feature = "vec2")]
            let q = self.nodes[node].kdcell.find_quadrant(pos);

            #[cfg(feature = "vec3")]
            let q = self.nodes[node].kdcell.find_octant(pos);

            node = self.nodes[node].children + q;
        }

        if self.nodes[node].is_empty() {
            self.nodes[node].pos = pos;
            self.nodes[node].mass = mass;
            return;
        }

        let p = self.nodes[node].pos;
        let m = self.nodes[node].mass;

        // Check if positions are equal (considering floating point precision)
        let diff = pos - p;
        if diff.norm_squared() < 1e-10 {
            self.nodes[node].mass += mass;
            return;
        }

        loop {
            let children = self.subdivide(node);

            #[cfg(feature = "vec2")]
            {
                let q1 = self.nodes[node].kdcell.find_quadrant(p);
                let q2 = self.nodes[node].kdcell.find_quadrant(pos);

                if q1 == q2 {
                    node = children + q1;
                } else {
                    let n1 = children + q1;
                    let n2 = children + q2;

                    self.nodes[n1].pos = p;
                    self.nodes[n1].mass = m;
                    self.nodes[n2].pos = pos;
                    self.nodes[n2].mass = mass;
                    return;
                }
            }

            #[cfg(feature = "vec3")]
            {
                let q1 = self.nodes[node].kdcell.find_octant(p);
                let q2 = self.nodes[node].kdcell.find_octant(pos);

                if q1 == q2 {
                    node = children + q1;
                } else {
                    let n1 = children + q1;
                    let n2 = children + q2;

                    self.nodes[n1].pos = p;
                    self.nodes[n1].mass = m;
                    self.nodes[n2].pos = pos;
                    self.nodes[n2].mass = mass;
                    return;
                }
            }
        }
    }

    /// @brief Propagates mass and center of mass up the tree.
    ///
    /// After all bodies are inserted, this method computes the total mass
    /// and center of mass for each branch node by aggregating its children.
    /// Must be called after all insertions and before force computation.
    pub fn propagate(&mut self) {
        for &node in self.parents.iter().rev() {
            let i = self.nodes[node].children;

            #[cfg(feature = "vec2")]
            {
                self.nodes[node].pos = self.nodes[i].pos * self.nodes[i].mass
                    + self.nodes[i + 1].pos * self.nodes[i + 1].mass
                    + self.nodes[i + 2].pos * self.nodes[i + 2].mass
                    + self.nodes[i + 3].pos * self.nodes[i + 3].mass;

                self.nodes[node].mass = self.nodes[i].mass
                    + self.nodes[i + 1].mass
                    + self.nodes[i + 2].mass
                    + self.nodes[i + 3].mass;
            }

            #[cfg(feature = "vec3")]
            {
                self.nodes[node].pos = self.nodes[i].pos * self.nodes[i].mass
                    + self.nodes[i + 1].pos * self.nodes[i + 1].mass
                    + self.nodes[i + 2].pos * self.nodes[i + 2].mass
                    + self.nodes[i + 3].pos * self.nodes[i + 3].mass
                    + self.nodes[i + 4].pos * self.nodes[i + 4].mass
                    + self.nodes[i + 5].pos * self.nodes[i + 5].mass
                    + self.nodes[i + 6].pos * self.nodes[i + 6].mass
                    + self.nodes[i + 7].pos * self.nodes[i + 7].mass;

                self.nodes[node].mass = self.nodes[i].mass
                    + self.nodes[i + 1].mass
                    + self.nodes[i + 2].mass
                    + self.nodes[i + 3].mass
                    + self.nodes[i + 4].mass
                    + self.nodes[i + 5].mass
                    + self.nodes[i + 6].mass
                    + self.nodes[i + 7].mass;
            }

            let mass = self.nodes[node].mass;
            if mass > 0.0 {
                self.nodes[node].pos /= mass;
            }
        }
    }

    /// @brief Computes acceleration on a body using Barnes-Hut approximation.
    ///
    /// Traverses the tree and computes gravitational acceleration.
    /// Uses approximation when a node is sufficiently far away (determined by theta).
    ///
    /// @param pos Position of the body to compute acceleration for
    /// @param theta Opening angle parameter (smaller = more accurate, slower)
    /// @param epsilon Softening parameter to avoid singularities
    /// @return Acceleration vector
    pub fn acc(&self, pos: Vector, theta: f32, epsilon: f32) -> Vector {
        let mut acc = Vector::zero();

        let t_sq = theta * theta;
        let e_sq = epsilon * epsilon;

        let mut node = Self::ROOT;
        loop {
            let n = &self.nodes[node];

            let d = n.pos - pos;
            let d_sq = d.norm_squared();

            if n.is_leaf() || n.kdcell.size * n.kdcell.size < d_sq * t_sq {
                let denom = (d_sq + e_sq) * d_sq.sqrt();
                if denom > 0.0 {
                    acc += d * (n.mass / denom).min(f32::MAX);
                }

                if n.next == 0 {
                    break;
                }
                node = n.next;
            } else {
                node = n.children;
            }
        }

        acc
    }

    /// @brief Returns a displayable wrapper that includes bodies.
    ///
    /// Creates a Display implementation that shows the tree structure
    /// along with the bodies contained in each node.
    ///
    /// @param bodies Slice of bodies to display with the tree
    /// @return A displayable wrapper
    pub fn display_with_bodies<'a>(&'a self, bodies: &'a [Body]) -> impl fmt::Display + 'a {
        struct Wrapper<'a> {
            tree: &'a KdTree,
            bodies: &'a [Body],
        }

        impl<'a> fmt::Display for Wrapper<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let tree = self.tree;
                let bodies = self.bodies;

                if tree.nodes.is_empty() {
                    writeln!(f, "<empty kdtree>")?;
                    return Ok(());
                }

                #[cfg(feature = "vec2")]
                writeln!(f, "KdTree (Quadtree, nodes: {}):", tree.nodes.len())?;

                #[cfg(feature = "vec3")]
                writeln!(f, "KdTree (Octree, nodes: {}):", tree.nodes.len())?;

                fn recurse(
                    f: &mut fmt::Formatter<'_>,
                    tree: &KdTree,
                    node_idx: usize,
                    bodies: &[Body],
                    prefix: &str,
                    is_last: bool,
                ) -> fmt::Result {
                    let node = &tree.nodes[node_idx];

                    let connector = if prefix.is_empty() {
                        ""
                    } else if is_last {
                        "└─ "
                    } else {
                        "├─ "
                    };

                    writeln!(
                        f,
                        "{}{}[{}] cell={} mass={:.6} pos={} {}",
                        prefix,
                        connector,
                        node_idx,
                        node.kdcell,
                        node.mass,
                        node.pos,
                        if node.is_leaf() {
                            if node.is_empty() {
                                "empty leaf"
                            } else {
                                "leaf"
                            }
                        } else {
                            "branch"
                        }
                    )?;

                    // Display bodies in leaf nodes
                    if node.is_leaf() {
                        let next_prefix = if prefix.is_empty() {
                            String::from("   ")
                        } else if is_last {
                            format!("{}   ", prefix)
                        } else {
                            format!("{}│  ", prefix)
                        };

                        let mut count = 0;
                        for body in bodies {
                            if node.kdcell.contains(body.pos) {
                                count += 1;
                                writeln!(f, "{}└─ {}", next_prefix, body)?;
                            }
                        }

                        if count == 0 && node.is_empty() {
                            writeln!(f, "{}└─ <no bodies>", next_prefix)?;
                        }
                    }

                    // Recursive exploration on branch
                    if node.is_branch() {
                        let base = node.children;
                        let child_prefix = if prefix.is_empty() {
                            String::new()
                        } else if is_last {
                            format!("{}   ", prefix)
                        } else {
                            format!("{}│  ", prefix)
                        };

                        for i in 0..KdTree::NUM_CHILDREN {
                            let idx = base + i;
                            if idx >= tree.nodes.len() {
                                writeln!(
                                    f,
                                    "{}{}[{}] <missing>",
                                    child_prefix,
                                    if i == KdTree::NUM_CHILDREN - 1 {
                                        "└─ "
                                    } else {
                                        "├─ "
                                    },
                                    idx
                                )?;
                                continue;
                            }
                            recurse(
                                f,
                                tree,
                                idx,
                                bodies,
                                &child_prefix,
                                i == KdTree::NUM_CHILDREN - 1,
                            )?;
                        }
                    }

                    Ok(())
                }

                recurse(f, tree, KdTree::ROOT, bodies, "", true)
            }
        }

        Wrapper { tree: self, bodies }
    }
}

/// @brief Display trait implementation for KdTree.
///
/// Provides a tree-like visualization of the structure without bodies.
impl fmt::Display for KdTree {
    /// @brief Formats the KdTree for display.
    ///
    /// @param f The formatter
    /// @return Result of the formatting operation
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nodes.is_empty() {
            writeln!(f, "<empty kdtree>")?;
            return Ok(());
        }

        #[cfg(feature = "vec2")]
        writeln!(f, "KdTree (Quadtree, nodes: {}):", self.nodes.len())?;

        #[cfg(feature = "vec3")]
        writeln!(f, "KdTree (Octree, nodes: {}):", self.nodes.len())?;

        fn recurse(
            f: &mut fmt::Formatter<'_>,
            tree: &KdTree,
            node_idx: usize,
            prefix: &str,
            is_last: bool,
        ) -> fmt::Result {
            let node = &tree.nodes[node_idx];

            let connector = if prefix.is_empty() {
                ""
            } else if is_last {
                "└─ "
            } else {
                "├─ "
            };

            writeln!(f, "{}{}{}", prefix, connector, node)?;

            // Recursive exploration on branch
            if node.is_branch() {
                let base = node.children;
                let child_prefix = if prefix.is_empty() {
                    String::new()
                } else if is_last {
                    format!("{}   ", prefix)
                } else {
                    format!("{}│  ", prefix)
                };

                for i in 0..KdTree::NUM_CHILDREN {
                    let idx = base + i;
                    if idx >= tree.nodes.len() {
                        writeln!(
                            f,
                            "{}{}[{}] <missing>",
                            child_prefix,
                            if i == KdTree::NUM_CHILDREN - 1 {
                                "└─ "
                            } else {
                                "├─ "
                            },
                            idx
                        )?;
                        continue;
                    }
                    recurse(f, tree, idx, &child_prefix, i == KdTree::NUM_CHILDREN - 1)?;
                }
            }

            Ok(())
        }

        recurse(f, self, Self::ROOT, "", true)
    }
}
