use super::{Node, Quad};
use crate::body::Body;
use crate::geom::Vec2;
use std::fmt;

#[derive(Debug)]
pub struct Quadtree {
    pub nodes: Vec<Node>,
    pub parents: Vec<usize>,
}

impl Quadtree {
    const ROOT: usize = 0;

    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            parents: Vec::new(),
        }
    }

    pub fn clear(&mut self, quad: Quad) {
        self.nodes.clear();
        self.parents.clear();
        self.nodes.push(Node::new(0, quad));
    }

    fn subdivide(&mut self, node: usize) -> usize {
        self.parents.push(node);
        let children = self.nodes.len();
        self.nodes[node].children = children;

        let nexts = [
            children + 1,
            children + 2,
            children + 3,
            self.nodes[node].next,
        ];
        let quads = self.nodes[node].quad.into_quadrants();
        for i in 0..4 {
            self.nodes.push(Node::new(nexts[i], quads[i]));
        }
        children
    }

    pub fn insert(&mut self, pos: Vec2, mass: f32) {
        let mut node = Self::ROOT;

        while self.nodes[node].is_branch() {
            let q = self.nodes[node].quad.find_quadrant(pos);
            node = self.nodes[node].children + q;
        }

        if self.nodes[node].is_empty() {
            self.nodes[node].pos = pos;
            self.nodes[node].mass = mass;
            return;
        }

        let p = self.nodes[node].pos;
        let m = self.nodes[node].mass;

        if pos == p {
            self.nodes[node].mass += mass;
            return;
        }

        loop {
            let children = self.subdivide(node);

            let q1 = self.nodes[node].quad.find_quadrant(p);
            let q2 = self.nodes[node].quad.find_quadrant(pos);

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

    pub fn propagate(&mut self) {
        for &node in self.parents.iter().rev() {
            let i = self.nodes[node].children;

            self.nodes[node].pos = self.nodes[i].pos * self.nodes[i].mass
                + self.nodes[i + 1].pos * self.nodes[i + 1].mass
                + self.nodes[i + 2].pos * self.nodes[i + 2].mass
                + self.nodes[i + 3].pos * self.nodes[i + 3].mass;

            self.nodes[node].mass = self.nodes[i].mass
                + self.nodes[i + 1].mass
                + self.nodes[i + 2].mass
                + self.nodes[i + 3].mass;

            let mass = self.nodes[node].mass;
            self.nodes[node].pos /= mass;
        }
    }

    pub fn acc(&self, pos: Vec2, theta: f32, epsilon: f32) -> Vec2 {
        let mut acc = Vec2::zero();

        let t_sq = theta * theta;
        let e_sq = epsilon * epsilon;

        let mut node = Self::ROOT;
        loop {
            let n = &self.nodes[node];

            let d = n.pos - pos;
            let d_sq = d.x * d.x + d.y * d.y;

            if n.is_leaf() || n.quad.size * n.quad.size < d_sq * t_sq {
                let denom = (d_sq + e_sq) * d_sq.sqrt();
                acc += d * (n.mass / denom).min(f32::MAX);

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

    /// Return a printable wrapper, including the bodies
    pub fn display_with_bodies<'a>(&'a self, bodies: &'a [Body]) -> impl fmt::Display + 'a {
        struct Wrapper<'a> {
            tree: &'a Quadtree,
            bodies: &'a [Body],
        }

        impl<'a> fmt::Display for Wrapper<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let tree = self.tree;
                let bodies = self.bodies;

                if tree.nodes.is_empty() {
                    writeln!(f, "<empty quadtree>")?;
                    return Ok(());
                }

                writeln!(f, "Quadtree (nodes: {}):", tree.nodes.len())?;

                fn recurse(
                    f: &mut fmt::Formatter<'_>,
                    tree: &Quadtree,
                    node_idx: usize,
                    bodies: &[Body],
                    prefix: &str,
                    is_last: bool,
                ) -> fmt::Result {
                    let node = &tree.nodes[node_idx];

                    let connector = if prefix.is_empty() {
                        "" // racine
                    } else if is_last {
                        "└─ "
                    } else {
                        "├─ "
                    };

                    writeln!(
                        f,
                        "{}{}[{}] quad(center={:.3},{:.3}, size={:.3}) mass={:.6} pos={:.3},{:.3} {}",
                        prefix,
                        connector,
                        node_idx,
                        node.quad.center.x,
                        node.quad.center.y,
                        node.quad.size,
                        node.mass,
                        node.pos.x,
                        node.pos.y,
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

                    // Si feuille, afficher bodies contenus
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
                            if node.quad.contains(body.pos) {
                                // Maybe &body.pos ??
                                count += 1;
                                writeln!(f, "{}└─ {}", next_prefix, body)?;
                            }
                        }

                        if count == 0 && node.is_empty() {
                            writeln!(f, "{}└─ <no bodies>", next_prefix)?;
                        }
                    }

                    // Si branche, descente récursive
                    if node.is_branch() {
                        let base = node.children;
                        let child_prefix = if prefix.is_empty() {
                            String::new()
                        } else if is_last {
                            format!("{}   ", prefix)
                        } else {
                            format!("{}│  ", prefix)
                        };

                        for i in 0..4 {
                            let idx = base + i;
                            if idx >= tree.nodes.len() {
                                writeln!(
                                    f,
                                    "{}{}[{}] <missing>",
                                    child_prefix,
                                    if i == 3 { "└─ " } else { "├─ " },
                                    idx
                                )?;
                                continue;
                            }
                            recurse(f, tree, idx, bodies, &child_prefix, i == 3)?;
                        }
                    }

                    Ok(())
                }

                recurse(f, tree, Quadtree::ROOT, bodies, "", true)
            }
        }

        Wrapper { tree: self, bodies }
    }
}

// impl fmt::Display for Quadtree {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         if self.nodes.is_empty() {
//             writeln!(f, "<empty quadtree>")?;
//             return Ok(());
//         }
//
//         writeln!(f, "Quadtree (nodes: {}):", self.nodes.len())?;
//
//         fn recurse(
//             f: &mut fmt::Formatter<'_>,
//             tree: &Quadtree,
//             node_idx: usize,
//             prefix: &str,
//             is_last: bool,
//         ) -> fmt::Result {
//             let node = &tree.nodes[node_idx];
//
//             let connector = if prefix.is_empty() {
//                 "" // Root
//             } else if is_last {
//                 "└─ "
//             } else {
//                 "├─ "
//             };
//
//             writeln!(
//                 f,
//                 "{}{}[{}] quad(center={:.3},{:.3}, size={:.3}) mass={:.6} pos={:.3},{:.3} {}",
//                 prefix,
//                 connector,
//                 node_idx,
//                 node.quad.center.x,
//                 node.quad.center.y,
//                 node.quad.size,
//                 node.mass,
//                 node.pos.x,
//                 node.pos.y,
//                 if node.is_leaf() {
//                     if node.is_empty() {
//                         "empty leaf"
//                     } else {
//                         "leaf"
//                     }
//                 } else {
//                     "branch"
//                 }
//             )?;
//
//             // Recursive exploration on branch
//             if node.is_branch() {
//                 let base = node.children;
//                 let child_prefix = if prefix.is_empty() {
//                     String::new()
//                 } else if is_last {
//                     format!("{}   ", prefix)
//                 } else {
//                     format!("{}│  ", prefix)
//                 };
//
//                 for i in 0..4 {
//                     let idx = base + i;
//                     if idx >= tree.nodes.len() {
//                         writeln!(
//                             f,
//                             "{}{}[{}] <missing>",
//                             child_prefix,
//                             if i == 3 { "└─ " } else { "├─ " },
//                             idx
//                         )?;
//                         continue;
//                     }
//                     recurse(f, tree, idx, &child_prefix, i == 3)?;
//                 }
//             }
//
//             Ok(())
//         }
//
//         recurse(f, self, Self::ROOT, "", true)
//     }
// }
