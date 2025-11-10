mod body;
mod cuda;
mod geom;
mod kdtree;
mod simul;

use crate::geom::Vec2;
use crate::simul::generate::generate_gaussian;

fn main() {
    let bodies = generate_gaussian(100, Vec2::new(0.0, 0.0), 10.0, 1.0, 0.5);
    for b in bodies.iter().take(50) {
        println!("{}", b);
    }
}
