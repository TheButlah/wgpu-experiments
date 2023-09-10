use bytemuck::{Pod, Zeroable};
use nalgebra::{geometry::Point3, UnitQuaternion};

#[derive(Pod, Zeroable, Debug)]
pub struct Instance {
	pub homog: nalgebra::geometry::IsometryMatrix3<f32>,
}
