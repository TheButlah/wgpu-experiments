use wgpu::util::DeviceExt;

pub struct Shape {
	pub width: u32,
	pub height: u32,
}

pub struct Tex2d {
	pub texture: wgpu::Texture,
	pub view: wgpu::TextureView,
	pub sampler: wgpu::Sampler,
}
impl Tex2d {
	/// The amount of multisampling
	const N_SAMPLES: u8 = 1;
	const VIEW_DIM: wgpu::TextureViewDimension = wgpu::TextureViewDimension::D2;

	pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
		device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("Texture Bind Group Layout"),
			entries: &[
				wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Texture {
						sample_type: wgpu::TextureSampleType::Float {
							filterable: true,
						},
						view_dimension: Self::VIEW_DIM,
						multisampled: Self::N_SAMPLES > 1,
					},
					// Not an array, so we use `None`
					count: None,
				},
				wgpu::BindGroupLayoutEntry {
					binding: 1,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
					count: None,
				},
			],
		})
	}

	pub fn new_from_img_bytes(
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		bytes: &[u8],
		label: Option<&str>,
	) -> Self {
		let img = image::load_from_memory(bytes).unwrap();
		Self::new_from_img(device, queue, label, img)
	}

	pub fn new_from_rgb8(
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		label: Option<&str>,
		bytes: &[u8],
		Shape { width, height }: Shape,
	) -> Self {
		let tex_size = wgpu::Extent3d {
			width,
			height,
			// We are not using an array of images, so its just 1
			depth_or_array_layers: 1,
		};
		let texture = device.create_texture_with_data(
			queue,
			&wgpu::TextureDescriptor {
				label,
				size: tex_size,
				mip_level_count: 1,
				sample_count: 1,
				dimension: Self::VIEW_DIM.compatible_texture_dimension(),
				format: wgpu::TextureFormat::Rgba8UnormSrgb,
				usage: wgpu::TextureUsages::TEXTURE_BINDING
					| wgpu::TextureUsages::COPY_DST,
				view_formats: &[],
			},
			bytes,
		);
		let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
		// NOTE: The tutorial does this one manually instead of default.
		let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

		Self {
			texture,
			view,
			sampler,
		}
	}

	pub fn new_from_img(
		device: &wgpu::Device,
		queue: &wgpu::Queue,
		label: Option<&str>,
		img: image::DynamicImage,
	) -> Self {
		let width = img.width();
		let height = img.height();
		let rgba = img.into_rgba8();
		Self::new_from_rgb8(device, queue, label, &rgba, Shape { width, height })
	}
}
