use color_eyre::{eyre::bail, eyre::eyre, eyre::WrapErr, Help, Result};
use instant::Instant;
use log::{debug, warn};
use nalgebra::geometry::Point3;
use nalgebra::point;
use std::fmt::Write;
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;
use winit::window::{CursorGrabMode, Window};
use winit_input_helper::WinitInputHelper;

use crate::camera::Camera;
use crate::tex2d::Tex2d;
use crate::vertex::{Pos, Uv, Vertex};

pub struct RenderState {
	// Fields dropped in order of declaration.
	// Surface must be dropped before window.
	surface: wgpu::Surface,
	window: Window,
	device: wgpu::Device,
	queue: wgpu::Queue,
	config: wgpu::SurfaceConfiguration,
	pipeline: wgpu::RenderPipeline,
	vtx_buf: wgpu::Buffer,
	idx_buf: wgpu::Buffer,
	num_indices: u32,
	diffuse_bind_group: wgpu::BindGroup,
	camera: Camera,
	camera_buf: wgpu::Buffer,
	camera_bind_group: wgpu::BindGroup,
	fps: f32,
	last_render: Instant,
	last_title: Instant,
	title: String,
}
impl RenderState {
	pub async fn new(window: Window) -> Result<Self> {
		let size = window.inner_size();
		let instance = make_instance();

		// Safety: we store both `window` and `surface` in `State` so we can be sure that `surface`
		// is dropped first.
		let surface = unsafe { instance.create_surface(&window) }?;
		let (device, queue, adapter) =
			make_device_queue_adapter(&instance, &surface).await?;
		let config = configure_surface(&surface, &adapter, &device, size);

		let cam_data = make_camera(&config, &device);
		let diffuse_data = make_diffuse_texture(&device, &queue);
		let vertex_data = make_vertex_data(&device);

		let pipeline = make_pipeline(
			&device,
			&diffuse_data.bind_group_layout,
			&cam_data.bind_group_layout,
			&config,
		);

		lock_cursor(&window);

		Ok(Self {
			surface,
			window,
			device,
			queue,
			config,
			pipeline,
			vtx_buf: vertex_data.vtx_buf,
			idx_buf: vertex_data.idx_buf,
			num_indices: vertex_data.idx_buf_len.try_into().unwrap(),
			diffuse_bind_group: diffuse_data.bind_group,
			camera: cam_data.camera,
			camera_buf: cam_data.buf,
			camera_bind_group: cam_data.bind_group,
			fps: 0.,
			last_render: Instant::now(),
			last_title: Instant::now(),
			title: String::new(),
		})
	}

	pub fn update(&mut self, input: &WinitInputHelper) {
		let (dx, dy) = input.mouse_diff();
		if dx != 0. && dy != 0. {
			log::info!("mouse diff: ({}, {})", dx, dy);
		}
		if input.mouse_pressed(0) {
			lock_cursor(&self.window);
		} else if input.cursor().is_none() {
			unlock_cursor(&self.window);
		}
		self.camera.update(input);
		self.queue.write_buffer(
			&self.camera_buf,
			0,
			bytemuck::cast_slice(&[self.camera.proj_view()]),
		);
	}

	pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
		self.update_fps();

		let output = self.surface.get_current_texture()?;
		let view = output
			.texture
			.create_view(&wgpu::TextureViewDescriptor::default());

		let mut encoder =
			self.device
				.create_command_encoder(&wgpu::CommandEncoderDescriptor {
					label: Some("Render Encoder"),
				});
		encode_render_commands(
			&mut encoder,
			&view,
			&self.pipeline,
			&self.vtx_buf,
			&self.idx_buf,
			&self.diffuse_bind_group,
			&self.camera_bind_group,
			self.num_indices,
		);
		let commands = encoder.finish();

		self.queue.submit([commands]);
		output.present();

		Ok(())
	}

	pub fn resize(&mut self, size: PhysicalSize<u32>) {
		if size.width == 0 && size.height == 0 {
			return;
		}
		self.config.width = size.width;
		self.config.height = size.height;
		self.surface.configure(&self.device, &self.config);
	}

	pub fn size(&self) -> PhysicalSize<u32> {
		PhysicalSize {
			width: self.config.width,
			height: self.config.height,
		}
	}

	fn update_fps(&mut self) {
		// Values closer to 1 weight new values more.
		const SMOOTHING_FACTOR: f32 = 0.2;
		let now = Instant::now();
		let elapsed = now - self.last_render;
		let new_fps = 1.0 / elapsed.as_secs_f32();
		self.fps = self.fps * (1.0 - SMOOTHING_FACTOR) + new_fps * SMOOTHING_FACTOR;
		self.last_render = now;

		if (now - self.last_title).as_millis() > 100 {
			self.title.clear();
			write!(&mut self.title, "FPS: {:.1}", self.fps).ok();
			self.window.set_title(&self.title);
			self.last_title = now;
		}
	}
}

// --------------------------
// ---- Helper Functions ----
// --------------------------

fn make_instance() -> wgpu::Instance {
	let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all());
	let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
		backends,
		dx12_shader_compiler: Default::default(),
	});

	debug!(
		"Available wgpu adapters: {:#?}",
		instance
			.enumerate_adapters(backends)
			.map(|a| a.get_info())
			.collect::<Vec<_>>()
	);
	instance
}

async fn make_device_queue_adapter(
	instance: &wgpu::Instance,
	surface: &wgpu::Surface,
) -> Result<(wgpu::Device, wgpu::Queue, wgpu::Adapter)> {
	let adapter = instance
		.request_adapter(&wgpu::RequestAdapterOptions {
			power_preference: wgpu::PowerPreference::LowPower,
			force_fallback_adapter: false,
			/// Surface that is required to be presentable with the requested adapter. This does not
			/// create the surface, only guarantees that the adapter can present to said surface.
			compatible_surface: Some(surface),
		})
		.await
		.ok_or(eyre!("Failed to get a wgpu Adapter"))?;
	if !adapter.is_surface_supported(surface) {
		bail!("Adapter does not support surface!");
	}
	debug!("Chosen adapter: {:#?}", adapter.get_info());

	let limits = if cfg!(target_arch = "wasm32") {
		wgpu::Limits::downlevel_webgl2_defaults()
	} else {
		wgpu::Limits::downlevel_defaults()
	};
	let desc = wgpu::DeviceDescriptor {
		label: None,
		features: wgpu::Features::empty(),
		limits,
	};
	let (device, queue) = adapter
		.request_device(&desc, None)
		.await
		.wrap_err("Failed to get wgpu Device")
		.with_note(|| format!("WGPU Adapter was: {:#?}", adapter.get_info()))?;

	Ok((device, queue, adapter))
}

fn configure_surface(
	surface: &wgpu::Surface,
	adapter: &wgpu::Adapter,
	device: &wgpu::Device,
	size: PhysicalSize<u32>,
) -> wgpu::SurfaceConfiguration {
	let config = {
		// NOTE: all capabilities have the most preferred option as the 0th element.
		let caps = surface.get_capabilities(adapter);
		let format = caps
			.formats
			.iter()
			.copied()
			.find(|f| f.is_srgb())
			.unwrap_or_else(|| {
				warn!("GPU doesn't support sRGB, colors might not be as expected!");
				caps.formats[0]
			});
		wgpu::SurfaceConfiguration {
			// This lets the texture write to the screen (?)
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			format,
			width: size.width,
			height: size.height,
			present_mode: caps.present_modes[0],
			alpha_mode: caps.alpha_modes[0],
			view_formats: vec![],
		}
	};
	surface.configure(device, &config);

	config
}

struct CameraData {
	camera: Camera,
	bind_group_layout: wgpu::BindGroupLayout,
	bind_group: wgpu::BindGroup,
	buf: wgpu::Buffer,
}
fn make_camera(
	config: &wgpu::SurfaceConfiguration,
	device: &wgpu::Device,
) -> CameraData {
	let camera = {
		// to_radians() wasn't const yet :(
		const FOVY: f32 = 45.0 / 180.0 * std::f32::consts::PI;
		const ZNEAR: f32 = 0.1;
		const ZFAR: f32 = 100.0;
		const EYE: Point3<f32> = point![0., 0., 1.];
		const ORIGIN: Point3<f32> = point![0., 0., 0.];
		Camera {
			pos: EYE,
			dir: ORIGIN - EYE,
			proj: nalgebra::geometry::Perspective3::new(
				config.width as f32 / config.height as f32,
				FOVY,
				ZNEAR,
				ZFAR,
			),
			speed: 0.2,
		}
	};
	let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("Camera Uniform"),
		usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		contents: bytemuck::cast_slice(&[camera.proj_view()]),
	});
	let bind_group_layout =
		device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("Camera Bind Group Layout"),
			entries: &[wgpu::BindGroupLayoutEntry {
				binding: 0,
				visibility: wgpu::ShaderStages::VERTEX,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Uniform,
					has_dynamic_offset: false,
					// We could specify this for more performance, but meh
					min_binding_size: None,
				},
				count: None,
			}],
		});
	let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
		label: Some("camera_bind_group"),
		layout: &bind_group_layout,
		entries: &[wgpu::BindGroupEntry {
			binding: 0,
			resource: buf.as_entire_binding(),
		}],
	});

	CameraData {
		camera,
		bind_group_layout,
		bind_group,
		buf,
	}
}

fn make_pipeline(
	device: &wgpu::Device,
	tex_bind_group_layout: &wgpu::BindGroupLayout,
	cam_bind_group_layout: &wgpu::BindGroupLayout,
	config: &wgpu::SurfaceConfiguration,
) -> wgpu::RenderPipeline {
	// Can also use `include_wgsl!()`
	let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
		label: Some("shader.wgsl"),
		source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
	});

	let pipeline_layout =
		device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("Render Pipeline Layout"),
			bind_group_layouts: &[tex_bind_group_layout, cam_bind_group_layout],
			push_constant_ranges: &[],
		});

	device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
		label: Some("Render Pipeline"),
		layout: Some(&pipeline_layout),
		vertex: wgpu::VertexState {
			module: &shader,
			entry_point: "vs_main",
			buffers: &[Vertex::vb_layout()],
		},
		fragment: Some(wgpu::FragmentState {
			module: &shader,
			entry_point: "fs_main",
			targets: &[Some(wgpu::ColorTargetState {
				// Shader texture format will be same as what we configured earlier
				format: config.format,
				// Blend will simply replace old pixel data with new
				blend: Some(wgpu::BlendState::REPLACE),
				// We are writing to all RGBA channels
				write_mask: wgpu::ColorWrites::ALL,
			})],
		}),
		primitive: wgpu::PrimitiveState {
			topology: wgpu::PrimitiveTopology::TriangleList,
			strip_index_format: None,
			front_face: wgpu::FrontFace::Ccw,
			cull_mode: Some(wgpu::Face::Back),
			// The next three avoid needing additional features
			unclipped_depth: false,
			polygon_mode: wgpu::PolygonMode::Fill,
			conservative: false,
		},
		depth_stencil: None,
		// We won't be using multisampling, so do 1x
		multisample: wgpu::MultisampleState {
			count: 1,
			mask: !0,
			alpha_to_coverage_enabled: false,
		},
		// I don't understand this one, but the tutorial set it to `None`
		multiview: None,
	})
}

struct DiffuseData {
	bind_group_layout: wgpu::BindGroupLayout,
	bind_group: wgpu::BindGroup,
}
fn make_diffuse_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> DiffuseData {
	let tex = Tex2d::new_from_img_bytes(
		device,
		queue,
		include_bytes!("tree.png"),
		Some("Diffuse Texture"),
	);
	let bind_group_layout = Tex2d::layout(device);
	let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
		label: Some("diffuse_bind_group"),
		layout: &bind_group_layout,
		entries: &[
			wgpu::BindGroupEntry {
				binding: 0,
				resource: wgpu::BindingResource::TextureView(&tex.view),
			},
			wgpu::BindGroupEntry {
				binding: 1,
				resource: wgpu::BindingResource::Sampler(&tex.sampler),
			},
		],
	});
	DiffuseData {
		bind_group_layout,
		bind_group,
	}
}

struct VertexData {
	vtx_buf: wgpu::Buffer,
	idx_buf: wgpu::Buffer,
	idx_buf_len: usize,
}
fn make_vertex_data(device: &wgpu::Device) -> VertexData {
	// Describes a square.
	const VERTICES: &[Vertex] = &[
		// Starts at top left of square, goes Ccw
		Vertex::new(Pos::new(-0.5, 0.5, 0.0), Uv { u: 0.0, v: 0.0 }),
		Vertex::new(Pos::new(-0.5, -0.5, 0.0), Uv { u: 0.0, v: 1.0 }),
		Vertex::new(Pos::new(0.5, -0.5, 0.0), Uv { u: 1.0, v: 1.0 }),
		Vertex::new(Pos::new(0.5, 0.5, 0.0), Uv { u: 1.0, v: 0.0 }),
	];

	const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

	let vtx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("Vertex Buffer"),
		contents: bytemuck::cast_slice(VERTICES),
		usage: wgpu::BufferUsages::VERTEX,
	});

	let idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
		label: Some("Index Buffer"),
		contents: bytemuck::cast_slice(INDICES),
		usage: wgpu::BufferUsages::INDEX,
	});
	let idx_buf_len = INDICES.len();

	VertexData {
		vtx_buf,
		idx_buf,
		idx_buf_len,
	}
}

#[allow(clippy::too_many_arguments)]
fn encode_render_commands(
	encoder: &mut wgpu::CommandEncoder,
	view: &wgpu::TextureView,
	pipeline: &wgpu::RenderPipeline,
	vtx_buf: &wgpu::Buffer,
	idx_buf: &wgpu::Buffer,
	diffuse_bind_group: &wgpu::BindGroup,
	camera_bind_group: &wgpu::BindGroup,
	num_indices: u32,
) {
	let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
		label: Some("Render Pass"),
		color_attachments: &[Some(wgpu::RenderPassColorAttachment {
			view,
			resolve_target: None,
			ops: wgpu::Operations {
				load: wgpu::LoadOp::Clear(wgpu::Color {
					r: 0.1,
					g: 0.2,
					b: 0.3,
					a: 1.0,
				}),
				store: true,
			},
		})],
		depth_stencil_attachment: None,
	});

	render_pass.set_pipeline(pipeline);

	render_pass.set_vertex_buffer(0, vtx_buf.slice(..));
	render_pass.set_index_buffer(idx_buf.slice(..), wgpu::IndexFormat::Uint16);

	render_pass.set_bind_group(0, diffuse_bind_group, &[]);
	render_pass.set_bind_group(1, camera_bind_group, &[]);
	// render_pass.draw(0..self.num_vertices, 0..1)
	render_pass.draw_indexed(0..num_indices, 0, 0..1)
}

fn lock_cursor(window: &Window) {
	window
		.set_cursor_grab(CursorGrabMode::Locked)
		.or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
		.unwrap();
	window.set_cursor_visible(false);
}

fn unlock_cursor(window: &Window) {
	window.set_cursor_visible(true);
	window.set_cursor_grab(CursorGrabMode::None).unwrap();
}
