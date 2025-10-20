use super::{Node, Quad};

#[derive(Debug)]
pub struct Quadtree {
    pub nodes: Vec<Node>,
}

impl Quadtree {
    pub fn new(root_quad: Quad) -> Self {
        let mut nodes = Vec::new();
        nodes.push(Node::new(root_quad));
        Self { nodes }
    }

    pub fn clear(&mut self, quad: Quad) {
        self.nodes.clear();
        self.nodes.push(Node::new(quad));
    }

    pub fn subdivide(&mut self, node: usize) -> usize {
        let children = self.nodes.len();
        self.nodes[node].children = children;

        let quads = self.nodes[node].quad.into_quadrants();
        for i in 0..4 {
            self.nodes.push(Node::new(quads[i]));
        }
        children
    }
}
