use crate::geom::vec2::Vec2;

#[derive(Clone, Copy, Debug)]
pub struct Quad {
    pub center: Vec2,
    pub size: f32,
}

impl Quad {
    pub fn new(center: Vec2, size: f32) -> Self {
        Self { center, size }
    }

    pub fn into_quadrant(mut self, i: usize) -> Self {
        self.size *= 0.5;
        self.center.x += (0.5 - (i & 1) as f32) * self.size;
        self.center.y += (0.5 - (i >> 1) as f32) * self.size;
        self
    }

    pub fn into_quadrants(&self) -> [Quad; 4] {
        [0, 1, 2, 3].map(|i| self.into_quadrant(i))
    }
}
