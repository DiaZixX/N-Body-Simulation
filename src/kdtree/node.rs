use super::quad::Quad;
use crate::geom::vec2::Vec2;

#[derive(Debug, Clone)]
pub struct Node {
    pub children: usize,
    pub quad: Quad,
    pub pos: Vec2,
    pub mass: f32,
}

impl Node {
    pub fn new(quad: Quad) -> Self {
        Self {
            children: 0,
            quad,
            pos: Vec2::zero(),
            mass: 0.0,
        }
    }
}
