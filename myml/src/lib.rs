mod array_util;

mod traits;
pub use traits::{ActivationFn, Layer, Loss};

mod model;
pub use model::Model;

pub mod loss;

pub mod layers;
