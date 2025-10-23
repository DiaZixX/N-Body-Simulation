use crate::body::Body;
use crate::geom::Vec2;

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

    pub fn find_quadrant(&self, pos: Vec2) -> usize {
        ((pos.y > self.center.y) as usize) << 1 | (pos.x > self.center.x) as usize
    }

    pub fn new_containing(bodies: &[Body]) -> Self {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for body in bodies {
            min_x = min_x.min(body.pos.x);
            min_y = min_y.min(body.pos.y);
            max_x = max_x.max(body.pos.x);
            max_y = max_y.max(body.pos.y);
        }

        let center = Vec2::new(min_x + max_x, min_y + max_y) * 0.5;
        let size = (max_x - min_x).max(max_y - min_y);

        Self { center, size }
    }

    pub fn contains(&self, pos: Vec2) -> bool {
        let diff = self.center - pos;
        diff.x.abs() <= self.size / 2. && diff.y.abs() <= self.size / 2.
    }
}
