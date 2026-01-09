//! @file kdtree_tests.rs
//! @brief Integration tests for KdTree spatial hierarchy

use n_body_simulation::{Body, KdCell, KdTree, Node, Vector};
// Assuming you export KdTree, KdCell, and Node from lib.rs
// If not, you'll need to add: pub use kdtree::{KdTree, KdCell, Node};

#[cfg(test)]
mod kdcell_tests {
    use super::*;

    #[test]
    fn test_kdcell_creation() {
        #[cfg(feature = "vec2")]
        {
            let center = Vector::new(10.0, 20.0);
            let cell = KdCell::new(center, 50.0);
            assert_eq!(cell.center.x, 10.0);
            assert_eq!(cell.center.y, 20.0);
            assert_eq!(cell.size, 50.0);
        }

        #[cfg(feature = "vec3")]
        {
            let center = Vector::new(10.0, 20.0, 30.0);
            let cell = KdCell::new(center, 50.0);
            assert_eq!(cell.center.x, 10.0);
            assert_eq!(cell.center.y, 20.0);
            assert_eq!(cell.center.z, 30.0);
            assert_eq!(cell.size, 50.0);
        }
    }

    #[test]
    fn test_kdcell_contains() {
        #[cfg(feature = "vec2")]
        {
            let cell = KdCell::new(Vector::new(0.0, 0.0), 10.0);

            assert!(cell.contains(Vector::new(0.0, 0.0)));
            assert!(cell.contains(Vector::new(5.0, 5.0)));
            assert!(cell.contains(Vector::new(-5.0, -5.0)));
            assert!(cell.contains(Vector::new(5.0, -5.0)));

            assert!(!cell.contains(Vector::new(6.0, 0.0)));
            assert!(!cell.contains(Vector::new(0.0, 6.0)));
            assert!(!cell.contains(Vector::new(10.0, 10.0)));
        }

        #[cfg(feature = "vec3")]
        {
            let cell = KdCell::new(Vector::new(0.0, 0.0, 0.0), 10.0);

            assert!(cell.contains(Vector::new(0.0, 0.0, 0.0)));
            assert!(cell.contains(Vector::new(5.0, 5.0, 5.0)));
            assert!(cell.contains(Vector::new(-5.0, -5.0, -5.0)));

            assert!(!cell.contains(Vector::new(6.0, 0.0, 0.0)));
            assert!(!cell.contains(Vector::new(0.0, 6.0, 0.0)));
            assert!(!cell.contains(Vector::new(0.0, 0.0, 6.0)));
        }
    }

    #[cfg(feature = "vec2")]
    #[test]
    fn test_kdcell_quadrants() {
        let cell = KdCell::new(Vector::new(0.0, 0.0), 10.0);
        let quads = cell.into_quadrants();

        assert_eq!(quads.len(), 4);

        // Check size is halved
        for quad in &quads {
            assert_eq!(quad.size, 5.0);
        }

        // Check quadrant positions (approximate)
        assert!(quads[0].center.x > 0.0 && quads[0].center.y > 0.0); // Bottom-right
        assert!(quads[1].center.x < 0.0 && quads[1].center.y > 0.0); // Bottom-left
        assert!(quads[2].center.x > 0.0 && quads[2].center.y < 0.0); // Top-right
        assert!(quads[3].center.x < 0.0 && quads[3].center.y < 0.0); // Top-left
    }

    #[cfg(feature = "vec3")]
    #[test]
    fn test_kdcell_octants() {
        let cell = KdCell::new(Vector::new(0.0, 0.0, 0.0), 10.0);
        let octants = cell.into_octants();

        assert_eq!(octants.len(), 8);

        // Check size is halved
        for octant in &octants {
            assert_eq!(octant.size, 5.0);
        }
    }

    #[cfg(feature = "vec2")]
    #[test]
    fn test_find_quadrant() {
        let cell = KdCell::new(Vector::new(0.0, 0.0), 10.0);

        assert_eq!(cell.find_quadrant(Vector::new(-1.0, -1.0)), 0); // Bottom-left
        assert_eq!(cell.find_quadrant(Vector::new(1.0, -1.0)), 1); // Bottom-right
        assert_eq!(cell.find_quadrant(Vector::new(-1.0, 1.0)), 2); // Top-left
        assert_eq!(cell.find_quadrant(Vector::new(1.0, 1.0)), 3); // Top-right
    }

    #[cfg(feature = "vec3")]
    #[test]
    fn test_find_octant() {
        let cell = KdCell::new(Vector::new(0.0, 0.0, 0.0), 10.0);

        // Test a few octants
        let octant = cell.find_octant(Vector::new(-1.0, -1.0, -1.0));
        assert_eq!(octant, 0);

        let octant = cell.find_octant(Vector::new(1.0, 1.0, 1.0));
        assert_eq!(octant, 7);
    }

    #[test]
    fn test_new_containing() {
        #[cfg(feature = "vec2")]
        {
            let bodies = vec![
                Body::new(Vector::new(0.0, 0.0), Vector::zero(), 1.0, 0.5),
                Body::new(Vector::new(10.0, 10.0), Vector::zero(), 1.0, 0.5),
                Body::new(Vector::new(-5.0, 5.0), Vector::zero(), 1.0, 0.5),
            ];

            let cell = KdCell::new_containing(&bodies);

            // Check that all bodies are contained
            for body in &bodies {
                assert!(cell.contains(body.pos));
            }

            // Check that center is approximately at (2.5, 5.0)
            assert!((cell.center.x - 2.5).abs() < 0.1);
            assert!((cell.center.y - 5.0).abs() < 0.1);
        }

        #[cfg(feature = "vec3")]
        {
            let bodies = vec![
                Body::new(Vector::new(0.0, 0.0, 0.0), Vector::zero(), 1.0, 0.5),
                Body::new(Vector::new(10.0, 10.0, 10.0), Vector::zero(), 1.0, 0.5),
                Body::new(Vector::new(-5.0, 5.0, 2.0), Vector::zero(), 1.0, 0.5),
            ];

            let cell = KdCell::new_containing(&bodies);

            // Check that all bodies are contained
            for body in &bodies {
                assert!(cell.contains(body.pos));
            }
        }
    }
}

#[cfg(test)]
mod node_tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        #[cfg(feature = "vec2")]
        {
            let cell = KdCell::new(Vector::new(0.0, 0.0), 10.0);
            let node = Node::new(0, cell);

            assert_eq!(node.children, 0);
            assert_eq!(node.next, 0);
            assert_eq!(node.mass, 0.0);
            assert!(node.is_leaf());
            assert!(node.is_empty());
            assert!(!node.is_branch());
        }

        #[cfg(feature = "vec3")]
        {
            let cell = KdCell::new(Vector::new(0.0, 0.0, 0.0), 10.0);
            let node = Node::new(0, cell);

            assert!(node.is_leaf());
            assert!(node.is_empty());
        }
    }

    #[test]
    fn test_node_states() {
        #[cfg(feature = "vec2")]
        {
            let cell = KdCell::new(Vector::new(0.0, 0.0), 10.0);
            let mut node = Node::new(0, cell);

            // Initially empty leaf
            assert!(node.is_leaf());
            assert!(node.is_empty());
            assert!(!node.is_branch());

            // Add mass -> non-empty leaf
            node.mass = 5.0;
            assert!(node.is_leaf());
            assert!(!node.is_empty());
            assert!(!node.is_branch());

            // Add children -> branch
            node.children = 1;
            assert!(!node.is_leaf());
            assert!(!node.is_empty());
            assert!(node.is_branch());
        }

        #[cfg(feature = "vec3")]
        {
            let cell = KdCell::new(Vector::new(0.0, 0.0, 0.0), 10.0);
            let mut node = Node::new(0, cell);

            node.mass = 5.0;
            assert!(!node.is_empty());

            node.children = 1;
            assert!(node.is_branch());
        }
    }
}

#[cfg(test)]
mod kdtree_tests {
    use super::*;

    #[test]
    fn test_kdtree_creation() {
        let tree = KdTree::new();
        assert_eq!(tree.nodes.len(), 0);
        assert_eq!(tree.parents.len(), 0);
    }

    #[test]
    fn test_kdtree_clear() {
        #[cfg(feature = "vec2")]
        {
            let mut tree = KdTree::new();
            let cell = KdCell::new(Vector::new(0.0, 0.0), 100.0);

            tree.clear(cell);

            assert_eq!(tree.nodes.len(), 1);
            assert_eq!(tree.parents.len(), 0);
            assert!(tree.nodes[0].is_leaf());
            assert!(tree.nodes[0].is_empty());
        }

        #[cfg(feature = "vec3")]
        {
            let mut tree = KdTree::new();
            let cell = KdCell::new(Vector::new(0.0, 0.0, 0.0), 100.0);

            tree.clear(cell);

            assert_eq!(tree.nodes.len(), 1);
        }
    }

    #[test]
    fn test_kdtree_single_insert() {
        #[cfg(feature = "vec2")]
        {
            let mut tree = KdTree::new();
            let cell = KdCell::new(Vector::new(0.0, 0.0), 100.0);
            tree.clear(cell);

            tree.insert(Vector::new(5.0, 5.0), 10.0);

            assert_eq!(tree.nodes.len(), 1);
            assert_eq!(tree.nodes[0].mass, 10.0);
            assert_eq!(tree.nodes[0].pos.x, 5.0);
            assert_eq!(tree.nodes[0].pos.y, 5.0);
            assert!(tree.nodes[0].is_leaf());
        }

        #[cfg(feature = "vec3")]
        {
            let mut tree = KdTree::new();
            let cell = KdCell::new(Vector::new(0.0, 0.0, 0.0), 100.0);
            tree.clear(cell);

            tree.insert(Vector::new(5.0, 5.0, 5.0), 10.0);

            assert_eq!(tree.nodes.len(), 1);
            assert_eq!(tree.nodes[0].mass, 10.0);
        }
    }

    #[test]
    fn test_kdtree_multiple_inserts_same_position() {
        #[cfg(feature = "vec2")]
        {
            let mut tree = KdTree::new();
            let cell = KdCell::new(Vector::new(0.0, 0.0), 100.0);
            tree.clear(cell);

            tree.insert(Vector::new(5.0, 5.0), 10.0);
            tree.insert(Vector::new(5.0, 5.0), 5.0);

            // Should still be one node with combined mass
            assert_eq!(tree.nodes.len(), 1);
            assert_eq!(tree.nodes[0].mass, 15.0);
        }

        #[cfg(feature = "vec3")]
        {
            let mut tree = KdTree::new();
            let cell = KdCell::new(Vector::new(0.0, 0.0, 0.0), 100.0);
            tree.clear(cell);

            tree.insert(Vector::new(5.0, 5.0, 5.0), 10.0);
            tree.insert(Vector::new(5.0, 5.0, 5.0), 5.0);

            assert_eq!(tree.nodes[0].mass, 15.0);
        }
    }

    #[cfg(feature = "vec2")]
    #[test]
    fn test_kdtree_subdivide() {
        let mut tree = KdTree::new();
        let cell = KdCell::new(Vector::new(0.0, 0.0), 100.0);
        tree.clear(cell);

        // Insert two bodies in different quadrants
        tree.insert(Vector::new(-10.0, -10.0), 5.0);
        tree.insert(Vector::new(10.0, 10.0), 5.0);

        // Should have subdivided: 1 root + 4 children = 5 nodes
        assert_eq!(tree.nodes.len(), 5);
        assert!(tree.nodes[0].is_branch());
    }

    #[cfg(feature = "vec3")]
    #[test]
    fn test_kdtree_subdivide_3d() {
        let mut tree = KdTree::new();
        let cell = KdCell::new(Vector::new(0.0, 0.0, 0.0), 100.0);
        tree.clear(cell);

        // Insert two bodies in different octants
        tree.insert(Vector::new(-10.0, -10.0, -10.0), 5.0);
        tree.insert(Vector::new(10.0, 10.0, 10.0), 5.0);

        // Should have subdivided: 1 root + 8 children = 9 nodes
        assert_eq!(tree.nodes.len(), 9);
        assert!(tree.nodes[0].is_branch());
    }

    #[test]
    fn test_kdtree_propagate() {
        #[cfg(feature = "vec2")]
        {
            let mut tree = KdTree::new();
            let cell = KdCell::new(Vector::new(0.0, 0.0), 100.0);
            tree.clear(cell);

            tree.insert(Vector::new(-10.0, 0.0), 5.0);
            tree.insert(Vector::new(10.0, 0.0), 5.0);
            tree.propagate();

            // Root should have total mass of 10.0
            assert_eq!(tree.nodes[0].mass, 10.0);

            // Center of mass should be at (0.0, 0.0)
            assert!((tree.nodes[0].pos.x - 0.0).abs() < 0.01);
            assert!((tree.nodes[0].pos.y - 0.0).abs() < 0.01);
        }

        #[cfg(feature = "vec3")]
        {
            let mut tree = KdTree::new();
            let cell = KdCell::new(Vector::new(0.0, 0.0, 0.0), 100.0);
            tree.clear(cell);

            tree.insert(Vector::new(-10.0, 0.0, 0.0), 5.0);
            tree.insert(Vector::new(10.0, 0.0, 0.0), 5.0);
            tree.propagate();

            assert_eq!(tree.nodes[0].mass, 10.0);
            assert!((tree.nodes[0].pos.x - 0.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_kdtree_acceleration() {
        #[cfg(feature = "vec2")]
        {
            let mut tree = KdTree::new();
            let cell = KdCell::new(Vector::new(0.0, 0.0), 100.0);
            tree.clear(cell);

            // Place a mass at (10, 0)
            tree.insert(Vector::new(10.0, 0.0), 100.0);
            tree.propagate();

            // Compute acceleration at origin
            let acc = tree.acc(Vector::new(0.0, 0.0), 0.5, 0.1);

            // Acceleration should point toward (10, 0), i.e., positive x
            assert!(acc.x > 0.0);
            assert!(acc.y.abs() < 0.01);
        }

        #[cfg(feature = "vec3")]
        {
            let mut tree = KdTree::new();
            let cell = KdCell::new(Vector::new(0.0, 0.0, 0.0), 100.0);
            tree.clear(cell);

            tree.insert(Vector::new(10.0, 0.0, 0.0), 100.0);
            tree.propagate();

            let acc = tree.acc(Vector::new(0.0, 0.0, 0.0), 0.5, 0.1);

            assert!(acc.x > 0.0);
            assert!(acc.y.abs() < 0.01);
            assert!(acc.z.abs() < 0.01);
        }
    }

    #[test]
    fn test_kdtree_complex_scenario() {
        #[cfg(feature = "vec2")]
        {
            let mut tree = KdTree::new();
            let cell = KdCell::new(Vector::new(0.0, 0.0), 100.0);
            tree.clear(cell);

            // Insert multiple bodies
            let positions = vec![
                (Vector::new(-20.0, -20.0), 10.0),
                (Vector::new(-20.0, 20.0), 15.0),
                (Vector::new(20.0, -20.0), 20.0),
                (Vector::new(20.0, 20.0), 25.0),
            ];

            for (pos, mass) in positions {
                tree.insert(pos, mass);
            }
            tree.propagate();

            // Root should have total mass
            assert_eq!(tree.nodes[0].mass, 70.0);

            // Test acceleration computation
            let acc = tree.acc(Vector::new(0.0, 0.0), 0.5, 1.0);

            // Acceleration should be non-zero
            assert!(acc.norm() > 0.0);
        }

        #[cfg(feature = "vec3")]
        {
            let mut tree = KdTree::new();
            let cell = KdCell::new(Vector::new(0.0, 0.0, 0.0), 100.0);
            tree.clear(cell);

            let positions = vec![
                (Vector::new(-20.0, -20.0, -20.0), 10.0),
                (Vector::new(20.0, 20.0, 20.0), 15.0),
            ];

            for (pos, mass) in positions {
                tree.insert(pos, mass);
            }
            tree.propagate();

            assert_eq!(tree.nodes[0].mass, 25.0);

            let acc = tree.acc(Vector::new(0.0, 0.0, 0.0), 0.5, 1.0);
            assert!(acc.norm() > 0.0);
        }
    }
}
