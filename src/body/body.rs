use crate::geom::vec2::Vec2;
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub pos: Vec2,
    pub vel: Vec2,
    pub acc: Vec2,
    pub mass: f32,
    /// Radius for collision
    pub radius: f32,
}

impl Body {
    /// Acceleration is initialized at zero
    pub fn new(pos: Vec2, vel: Vec2, mass: f32, radius: f32) -> Self {
        Self {
            pos,
            vel,
            acc: Vec2::zero(),
            mass,
            radius,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.vel += self.acc * dt;
        self.pos += self.vel * dt;
    }

    /// Reinit the acceleration to zero -> used to recompute forces
    pub fn reset_acceleration(&mut self) {
        self.acc = Vec2::zero();
    }
}

impl fmt::Display for Body {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Body(pos=({}, {}), vel=({}, {}), mass={:.2}, radius={:.2})",
            self.pos.x, self.pos.y, self.vel.x, self.vel.y, self.mass, self.radius
        )
    }
}
