use nalgebra::geometry::{IsometryMatrix3, Perspective3};
use nalgebra::{
	matrix, Matrix4, Point3, Translation3, UnitQuaternion, Vector3,
};
use winit::event::VirtualKeyCode;
use winit_input_helper::WinitInputHelper;

const UP: Vector3<f32> = Vector3::new(0., 1., 0.);

/// OpenGL convention (which nalgebra follows): z goes from [-1, 1].
/// WebGPU uses [0, 1] for z.
const OPENGL_TO_WGPU_M: Matrix4<f32> = matrix![
	1.0, 0.0, 0.0, 0.0;
	0.0, 1.0, 0.0, 0.0;
	0.0, 0.0, 0.5, 0.0;
	0.0, 0.0, 0.5, 1.0;
];

pub struct Camera {
	pub pos: Point3<f32>,
	pub dir: Vector3<f32>,
	pub proj: Perspective3<f32>,
	pub speed: f32,
}
impl Camera {
	/// # Arguments
	/// - `cam_t`: The isometry of the camera, with respect to world
	pub fn proj_view(&self) -> Matrix4<f32> {
		let target = self.pos + self.dir;
		let view = IsometryMatrix3::look_at_rh(&self.pos, &target, &UP);
		OPENGL_TO_WGPU_M * self.proj.as_matrix() * view.to_matrix()
	}

	pub fn update(&mut self, input: &WinitInputHelper) {
		self.dir = delta_rot(self.speed, input) * self.dir;
		self.pos = delta_pos(self.speed, &self.dir, input) * self.pos;
	}
}

fn delta_pos(
	speed: f32,
	forward_dir: &Vector3<f32>,
	input: &WinitInputHelper,
) -> Translation3<f32> {
	let right_dir = forward_dir.cross(&UP);

	#[inline]
	fn sign(pos: bool, neg: bool) -> f32 {
		if pos {
			1.
		} else if neg {
			-1.
		} else {
			0.
		}
	}

	use VirtualKeyCode as K;
	let mut delta = sign(input.key_held(K::W), input.key_held(K::S)) * forward_dir;
	delta += sign(input.key_held(K::D), input.key_held(K::A)) * right_dir;
	delta += sign(input.key_held(K::E), input.key_held(K::Q)) * UP;
	Translation3::from(speed * delta)
}

fn delta_rot(sensitivity: f32, input: &WinitInputHelper) -> UnitQuaternion<f32> {
	let sensitivity = sensitivity * 0.01;
	// TODO
	let (dx, dy) = input.mouse_diff();
	let dx = dx * sensitivity;
	let dy = dy * sensitivity;

	UnitQuaternion::from_euler_angles(-dy, -dx, 0.)
}
