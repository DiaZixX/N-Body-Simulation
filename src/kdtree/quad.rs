use n_body_simulation::{Body, Vector};

#[derive(Clone, Copy, Debug)]
pub struct Quad {
    pub center: Vector,
    pub size: f32,
}

impl Quad {
    pub fn new(center: Vector, size: f32) -> Self {
        Self { center, size }
    }

    #[cfg(feature = "vec2")]
    pub fn into_quadrant(mut self, i: usize) -> Self {
        self.size *= 0.5;
        self.center.x += (0.5 - (i & 1) as f32) * self.size;
        self.center.y += (0.5 - (i >> 1) as f32) * self.size;
        self
    }

    #[cfg(feature = "vec3")]
    pub fn into_octant(mut self, i: usize) -> Self {
        self.size *= 0.5;
        self.center.x += (0.5 - (i & 1) as f32) * self.size;
        self.center.y += (0.5 - ((i >> 1) & 1) as f32) * self.size;
        self.center.z += (0.5 - (i >> 2) as f32) * self.size;
        self
    }

    #[cfg(feature = "vec2")]
    pub fn into_quadrants(&self) -> [Quad; 4] {
        [0, 1, 2, 3].map(|i| self.into_quadrant(i))
    }

    #[cfg(feature = "vec3")]
    pub fn into_octants(&self) -> [Quad; 8] {
        [0, 1, 2, 3, 4, 5, 6, 7].map(|i| self.into_octant(i))
    }

    #[cfg(feature = "vec2")]
    pub fn find_quadrant(&self, pos: Vector) -> usize {
        ((pos.y > self.center.y) as usize) << 1 | (pos.x > self.center.x) as usize
    }

    #[cfg(feature = "vec3")]
    pub fn find_octant(&self, pos: Vector) -> usize {
        ((pos.z > self.center.z) as usize) << 2
            | ((pos.y > self.center.y) as usize) << 1
            | (pos.x > self.center.x) as usize
    }

    pub fn new_containing(bodies: &[Body]) -> Self {
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        #[cfg(feature = "vec3")]
        let mut min_z = f32::MAX;
        #[cfg(feature = "vec3")]
        let mut max_z = f32::MIN;

        for body in bodies {
            min_x = min_x.min(body.pos.x);
            min_y = min_y.min(body.pos.y);
            max_x = max_x.max(body.pos.x);
            max_y = max_y.max(body.pos.y);

            #[cfg(feature = "vec3")]
            {
                min_z = min_z.min(body.pos.z);
                max_z = max_z.max(body.pos.z);
            }
        }

        let center = {
            #[cfg(feature = "vec2")]
            {
                Vector::new(min_x + max_x, min_y + max_y) * 0.5
            }

            #[cfg(feature = "vec3")]
            {
                Vector::new(min_x + max_x, min_y + max_y, min_z + max_z) * 0.5
            }
        };

        let size = {
            #[cfg(feature = "vec2")]
            {
                (max_x - min_x).max(max_y - min_y)
            }

            #[cfg(feature = "vec3")]
            {
                (max_x - min_x).max(max_y - min_y).max(max_z - min_z)
            }
        };

        Self { center, size }
    }

    pub fn contains(&self, pos: Vector) -> bool {
        let diff = self.center - pos;

        #[cfg(feature = "vec2")]
        {
            diff.x.abs() <= self.size / 2.0 && diff.y.abs() <= self.size / 2.0
        }

        #[cfg(feature = "vec3")]
        {
            diff.x.abs() <= self.size / 2.0
                && diff.y.abs() <= self.size / 2.0
                && diff.z.abs() <= self.size / 2.0
        }
    }
}
