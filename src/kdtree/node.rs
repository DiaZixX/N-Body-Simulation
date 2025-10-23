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
