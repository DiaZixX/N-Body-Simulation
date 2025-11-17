mod body;
mod cuda;
mod geom;
mod kdtree;
mod render;
mod simul;

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use crate::body::Body;
use crate::geom::Vec2;
use crate::render::renderer::Renderer;
use crate::simul::generate::generate_gaussian;

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    bodies: Vec<Body>,
}

impl ApplicationHandler for App {
    // Window creation & bodies initialization
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("N-Body simulation")
                .with_inner_size(winit::dpi::LogicalSize::new(800, 600));

            let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
            self.window = Some(window.clone());

            let bodies = generate_gaussian(200, Vec2::new(0.0, 0.0), 0.2, 1.0, 0.01);

            self.bodies = bodies.clone();
            let renderer = pollster::block_on(Renderer::new(window, &bodies));
            self.renderer = Some(renderer);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("Fermeture de la fenêtre...");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &self.renderer {
                    renderer.render();
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    // Bodies generation
    let bodies = generate_gaussian(100, Vec2::new(0.0, 0.0), 10.0, 1.0, 0.5);
    for b in bodies.iter().take(50) {
        println!("{}", b);
    }

    // Window initialization
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        window: None,
        renderer: None,
        bodies: vec![],
    };

    event_loop.run_app(&mut app).unwrap();
}
