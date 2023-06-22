mod camera;
mod render_state;
mod tex2d;
mod vertex;

use cfg_if::cfg_if;
use color_eyre::{eyre::WrapErr, Result};
use log::error;
use log::{info, warn};
use winit::event::VirtualKeyCode;
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::render_state::RenderState;

pub async fn run() -> Result<()> {
	color_eyre::install()?;
	cfg_if! {
		if #[cfg(target_arch = "wasm32")] {
			std::panic::set_hook(Box::new(console_error_panic_hook::hook));
			console_log::init_with_level(log::Level::Debug)
				.expect("Couldn't initialize logger");
		} else {
			use env_logger::Env;
			let env = Env::default().default_filter_or("wgpu_experiments=debug");
			env_logger::Builder::from_env(env).init();
		}
	}

	let event_loop = EventLoop::new();
	let window = WindowBuilder::new().build(&event_loop).unwrap();

	#[cfg(target_arch = "wasm32")]
	{
		// Winit prevents sizing with CSS, so we have to set
		// the size manually when on web.
		use winit::dpi::PhysicalSize;

		use winit::platform::web::WindowExtWebSys;
		web_sys::window()
			.and_then(|win| win.document())
			.and_then(|doc| {
				let parent = doc.get_element_by_id("wgpu-parent")?;
				let width = parent.client_width() as u32;
				let height = parent.client_height() as u32;
				info!("width: {}, height: {}", width, height);

				window.set_inner_size(PhysicalSize::new(width, height));

				let canvas = window.canvas();
				let style = canvas.style();
				style.remove_property("width").unwrap();
				style.remove_property("height").unwrap();

				parent.append_child(&canvas).ok()?;

				Some(())
			})
			.expect("Couldn't append canvas to document body.");
	}

	let mut input = WinitInputHelper::new();
	let mut state = RenderState::new(window)
		.await
		.wrap_err("Error when initializing wgpu state")?;

	info!("Starting event loop");
	event_loop.run(move |event, _e_loop, control_flow| {
		// When true, input_helper is done processing events.
		if !input.update(&event) {
			return;
		}

		// Handle close events
		{
			if input.key_pressed(VirtualKeyCode::Escape)
				|| input.close_requested()
				|| input.destroyed()
			{
				info!("Close Requested");
				*control_flow = ControlFlow::Exit;
				return;
			}
		}

		if let Some(size) = input.window_resized() {
			state.resize(size);
		}

		state.update(&input);

		use wgpu::SurfaceError as E;
		match state.render() {
			Ok(_) => {}
			Err(E::Lost) => state.resize(state.size()),
			Err(E::OutOfMemory) => {
				error!("Out of memory!");
				*control_flow = ControlFlow::Exit;
			}
			Err(err) => {
				warn!("Error in event loop: {:?}", err);
			}
		}
	})
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub async fn wasm_start() -> Result<(), JsError> {
	run().await.map_err(|e| {
		let e: &(dyn std::error::Error + Send + Sync + 'static) = e.as_ref();
		JsError::from(e)
		// let b: Box<dyn std::error::Error + 'static> = e.into();
		// JsError::from(b)
	})
}
